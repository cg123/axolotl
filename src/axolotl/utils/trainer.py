"""Module containing the Trainer class and related functions"""

import json
import math
import os
import random
from contextlib import contextmanager
from functools import partial
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.cuda
from accelerate.logging import get_logger
from datasets import IterableDataset, disable_caching, enable_caching
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers.utils import is_torch_bf16_gpu_available

from axolotl.core.trainer_builder import HFCausalTrainerBuilder, HFRLTrainerBuilder
from axolotl.monkeypatch.trainer_eval_guard import patch_evaluation_loop_for_fsdp2
from axolotl.utils.distributed import reduce_and_broadcast
from axolotl.utils.environment import check_cuda_p2p_ib_support
from axolotl.utils.samplers import MultipackBatchSampler, get_dataset_lengths

LOG = get_logger("axolotl")


@torch.jit.script
def weighted_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor
):
    # Flatten the logits, labels, and weights tensors
    logits = logits.view(
        -1, logits.size(-1)
    )  # logits becomes of shape [batch_size*sequence_length, vocab_size]
    labels = labels.view(-1)  # labels becomes of shape [batch_size*sequence_length]
    weights = weights.view(-1)  # weights becomes of shape [batch_size*sequence_length]

    # Compute the unweighted cross entropy loss
    losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")

    # Apply the weights to the losses and compute their sum
    return (weights * losses).sum()


@torch.jit.script
def create_weighted_mask(labels: torch.Tensor):
    # Check if the tensor is 2D. If not, unsqueeze it to make it 2D
    if len(labels.shape) == 1:
        labels = labels.unsqueeze(0)

    weights = torch.zeros_like(labels).float()
    for i in range(labels.shape[0]):
        mask = labels[i] != -100

        # Create a tensor to track group ids
        group_ids = torch.zeros_like(labels[i]).int()
        curr_group_id = 0

        for j in range(1, len(labels[i])):
            if mask[j] and not mask[j - 1]:  # switch from masked to unmasked label
                curr_group_id += 1  # start new group
            group_ids[j] = (
                curr_group_id if mask[j] else 0
            )  # assign group id if unmasked label

        # Count only unmasked labels in each group
        group_counts = torch.bincount(group_ids[mask])

        mask_weights = torch.zeros_like(labels[i]).float()
        mask_weights[mask] = 1.0 / group_counts[group_ids[mask]]

        weights[i] = mask_weights

    return weights.squeeze()  # squeeze the output to match the input dimension


def trainer_weighted_loss(model_output, labels, shift_labels=True):
    logits = (
        model_output["logits"] if isinstance(model_output, dict) else model_output[0]
    )
    if shift_labels:
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

    weights = create_weighted_mask(labels)
    return weighted_cross_entropy(logits, labels, weights)


@contextmanager
def disable_datasets_caching():
    try:
        disable_caching()
        yield
    finally:
        enable_caching()


def add_position_ids(sample):
    """
    Handle both single-example and batched data.
    - single example: sample['input_ids'] is a list[int]
    - batched data: sample['input_ids'] is a list[list[int]]
    """
    # Return sample unchanged if "input_ids" is not present, or is empty
    if "input_ids" not in sample or not sample["input_ids"]:
        return sample

    input_ids = sample["input_ids"]

    # If first element is an int, it’s a single example
    # If first element is a list, it’s a batch
    if isinstance(input_ids[0], int):
        # ---- SINGLE EXAMPLE ----
        seq_len = len(input_ids)
        # Position IDs for a single example
        # As a list
        sample["position_ids"] = list(range(seq_len))
        sample["length"] = seq_len

    else:
        # ---- BATCHED EXAMPLES ----
        # input_ids is a list of lists
        position_ids_batch = []
        lengths_batch = []
        for seq in input_ids:
            seq_len = len(seq)
            position_ids_batch.append(list(range(seq_len)))
            lengths_batch.append(seq_len)

        # Now store them back
        sample["position_ids"] = position_ids_batch
        sample["length"] = lengths_batch

    return sample


def _determine_chunk_boundaries(
    sample_len: int,
    input_ids_list: List[int],
    num_chunks_target: int,
    min_chunk_len: int,
    split_on_token_ids: Optional[List[int]],
) -> List[int]:
    """
    Calculates the end indices of chunks.
    Returns a sorted list of unique chunk end indices (exclusive).
    Fewer chunks than num_chunks_target may be formed if constraints aren't met.
    """
    if sample_len == 0:
        return []

    # Identify preferred split locations (index *after* the special token)
    preferred_splits_set: Set[int] = set()
    if split_on_token_ids:
        for i, token_id in enumerate(input_ids_list):
            # A split occurs *after* the token. i+1 is the end of the current chunk.
            # It must be < sample_len to be an internal split point.
            if token_id in split_on_token_ids and (i + 1) < sample_len:
                preferred_splits_set.add(i + 1)
    
    # Sort preferred splits for consistent selection if multiple are eligible in a range
    sorted_preferred_splits = sorted(list(preferred_splits_set))

    # If only one chunk is target, or sample is too short for even one min_chunk_len,
    # the whole sample becomes a single chunk.
    # The loop below will also handle cases where num_chunks_target > 1 but sample is too short.
    if num_chunks_target == 1 or sample_len < min_chunk_len :
        return [sample_len]

    # --- Multi-chunk logic ---
    # internal_split_points stores the end indices of the first (N-1) chunks.
    internal_split_points: List[int] = []
    current_pos_in_sample = 0  # Start of the segment from which the current chunk is being formed.

    # Attempt to define (num_chunks_target - 1) split points for num_chunks_target chunks
    for _ in range(num_chunks_target - 1):
        # Number of chunks that still need to be formed from current_pos_in_sample onwards
        # This includes the one for which we are defining an end, plus subsequent ones.
        num_chunks_to_form_from_here = num_chunks_target - len(internal_split_points)

        # Min length required from current_pos_in_sample for all remaining chunks
        min_len_needed_for_all_remaining = min_chunk_len * num_chunks_to_form_from_here
        
        if (sample_len - current_pos_in_sample) < min_len_needed_for_all_remaining:
            break  # Not enough total length left

        # Determine valid range for the *end position* of the *current* chunk
        # Min end: current chunk takes at least min_chunk_len
        min_end_for_this_chunk = current_pos_in_sample + min_chunk_len
        # Max end: leave enough space for subsequent (num_chunks_to_form_from_here - 1) chunks
        # to each have min_chunk_len
        max_end_for_this_chunk = sample_len - (min_chunk_len * (num_chunks_to_form_from_here - 1))
        
        if min_end_for_this_chunk > max_end_for_this_chunk: # Should be caught by above check, but as safeguard
            break 
        
        # Find preferred split locations within the valid range for this chunk's end
        eligible_preferred = [
            p for p in sorted_preferred_splits
            if min_end_for_this_chunk <= p <= max_end_for_this_chunk and \
               p > current_pos_in_sample # Must advance position
        ]
        
        chosen_split_point = -1
        if eligible_preferred:
            chosen_split_point = random.choice(eligible_preferred)
        else:
            # If no (eligible) preferred splits, pick randomly.
            # random.randint requires start <= end, ensured by min_end <= max_end check above.
            chosen_split_point = random.randint(min_end_for_this_chunk, max_end_for_this_chunk)
        
        if chosen_split_point <= current_pos_in_sample: # Fallback if split doesn't advance
            break

        internal_split_points.append(chosen_split_point)
        current_pos_in_sample = chosen_split_point

        if current_pos_in_sample >= sample_len: # Reached end of sample
            break
    
    # --- Finalize chunk boundaries ---
    # `internal_split_points` contains s1, s2, ... The full list of chunk end points
    # should effectively partition the sample, ending with sample_len.
    final_chunk_ends = list(internal_split_points) # Make a copy

    # Ensure sample_len is the ultimate boundary if there's content.
    if sample_len > 0:
        if not final_chunk_ends or final_chunk_ends[-1] < sample_len:
            final_chunk_ends.append(sample_len)
    
    # Clean up: ensure uniqueness and sort. Filter out invalid values (e.g., 0 or > sample_len).
    if not final_chunk_ends:
        return []

    unique_indices = set()
    for idx in final_chunk_ends:
        if 0 < idx <= sample_len: # Ensure valid indices
            unique_indices.add(idx)
    
    return sorted(list(unique_indices))


def add_pose_position_ids(
    sample: dict,
    max_context_len: int = 32768,
    split_on_token_ids: Optional[List[int]] = None,
    num_chunks: int = 2,
    min_chunk_len: int = 10,
    pose_probability: float = 1.0,
):
    """
    Applies Positional Skip-wisE (PoSE) to extend context length by manipulating position_ids.
    Unifies random chunking with token-ID-guided splitting.
    'split_on_token_ids' provides preferred split locations if available within constraints.
    'num_chunks' is the target; fewer may be created if sample_len is too short.
    """
    input_ids_val = sample["input_ids"]
    input_ids_list = input_ids_val.tolist() if isinstance(input_ids_val, torch.Tensor) else input_ids_val
    
    sample_len = len(input_ids_list)

    if random.random() > pose_probability or sample_len == 0:
        sample["position_ids"] = torch.arange(sample_len, dtype=torch.long)
        sample["length"] = sample_len
        sample["sequence_len"] = sample_len
        return sample

    if num_chunks <= 0: # Target at least one chunk (the whole sequence)
        raise ValueError("Number of chunks (num_chunks) must be at least 1.")
    if min_chunk_len <= 0:
        raise ValueError("Minimum chunk length (min_chunk_len) must be positive.")

    # --- 1. Determine Chunk Boundaries ---
    # These are end indices (exclusive) of chunks in the original input.
    chunk_end_indices = _determine_chunk_boundaries(
        sample_len, input_ids_list, num_chunks, min_chunk_len, split_on_token_ids
    )

    if not chunk_end_indices: # No valid chunking possible (e.g. sample_len became 0 effectively) or sample_len was 0
        sample["position_ids"] = torch.arange(sample_len, dtype=torch.long) # Fallback to standard
        sample["length"] = sample_len
        sample["sequence_len"] = sample_len
        return sample

    # --- 2. Generate PoSE Position IDs based on Chunks ---
    final_pose_positions = []
    # Create (start, end) pairs for chunks. Start is 0 for the first chunk.
    # chunk_end_indices = [end1, end2, ..., sample_len]
    # boundaries_for_iter = [0, end1, end2, ..., sample_len]
    current_chunk_start_idx = 0
    
    max_skips_budget = max(0, max_context_len - sample_len)
    cumulative_skip = 0

    for i, current_chunk_end_idx in enumerate(chunk_end_indices):
        if current_chunk_start_idx >= current_chunk_end_idx: # Skip empty or invalid chunk segments
            current_chunk_start_idx = current_chunk_end_idx # Ensure progress for next iteration
            continue

        # Add skips
        if i > 0:
            skip_increment = 0
            if max_skips_budget > 0:
                skip_increment = random.randint(0, max_skips_budget) 
                max_skips_budget -= skip_increment
            cumulative_skip += skip_increment
        
        for original_pos in range(current_chunk_start_idx, current_chunk_end_idx):
            final_pose_positions.append(original_pos + cumulative_skip)
        
        current_chunk_start_idx = current_chunk_end_idx

    sample["position_ids"] = torch.tensor(final_pose_positions, dtype=torch.long)
    sample["length"] = len(final_pose_positions)

    if final_pose_positions:
        sample["sequence_len"] = final_pose_positions[-1] + 1
    else: # Should not happen if sample_len > 0 and chunk_end_indices is not empty
        sample["sequence_len"] = 0

    if len(final_pose_positions) != sample_len:
        raise AssertionError(
            f"PoSE Position IDs length {len(final_pose_positions)} "
            f"does not match input_ids length {sample_len}. "
            f"Generated chunk_ends: {chunk_end_indices}, "
            f"Final PoSE positions (first 10): {final_pose_positions[:10]}"
        )
    return sample

def add_length(sample):
    sample["length"] = len(sample["input_ids"])
    return sample


def drop_long_seq(sample, sequence_len=2048, min_sequence_len=2):
    """
    Drop samples whose sequence length is either too long (> sequence_len)
    or too short (< min_sequence_len).

    Works for both single-example (list[int]) or batched (list[list[int]]).
    """
    min_sequence_len = min_sequence_len or 2

    input_ids = sample["input_ids"]

    # Edge case: if input_ids is empty
    if not input_ids:
        # Decide if you want to drop or keep empty. Let's drop.
        return False

    # Check if single example or batched by looking at the first element
    if isinstance(input_ids[0], int):
        # Single example (input_ids is a list of int)
        length = len(input_ids)
        return min_sequence_len <= length <= sequence_len

    # Batched (input_ids is a list of lists)
    results = []
    for seq in input_ids:
        length = len(seq)
        results.append(min_sequence_len <= length <= sequence_len)
    return results


def process_datasets_for_packing(cfg, train_dataset, eval_dataset):
    drop_attn_mask = cfg.model_config_type in ["mamba", "gemma3"]
    if drop_attn_mask:
        LOG.info("dropping attention_mask column")
        train_dataset = train_dataset.remove_columns("attention_mask")
        if eval_dataset:
            eval_dataset = eval_dataset.remove_columns("attention_mask")

    if cfg.model_config_type in ["falcon", "mistral"]:
        LOG.info("dropping token_type_ids column if it exists")
        if "token_type_ids" in train_dataset.column_names:
            train_dataset = train_dataset.remove_columns("token_type_ids")
        if eval_dataset and "token_type_ids" in eval_dataset.column_names:
            eval_dataset = eval_dataset.remove_columns("token_type_ids")

    def drop_no_trainable_tokens(sample):
        """
        Drop samples if all labels are -100 (i.e., zero trainable tokens).
        Works for both single-example or batched input.
        """
        labels = sample["labels"]
        if not labels:
            return True

        # Check if single example or batch
        # If first element is an int, we assume a single example
        # If it's a list, we assume we're dealing with a batch
        if isinstance(labels[0], int):
            # Single example: return a single bool
            return np.any(labels != -100)

        # Batched: 'labels' is a list of lists
        # Return a list of booleans, one per sub-list
        results = [np.any(row_labels != -100) for row_labels in labels]
        return results

    try:
        prior_len = len(train_dataset)
    except TypeError:
        # handle iterable datasets case
        prior_len = None
    filter_map_kwargs = {}
    if not isinstance(train_dataset, IterableDataset):
        filter_map_kwargs["num_proc"] = cfg.dataset_processes
        filter_map_kwargs["load_from_cache_file"] = not cfg.is_preprocess

    drop_long_kwargs = {}
    if filter_map_kwargs:
        drop_long_kwargs["desc"] = "Drop Samples with Zero Trainable Tokens"
    train_dataset = train_dataset.filter(
        drop_no_trainable_tokens,
        batched=True,
        **filter_map_kwargs,
        **drop_long_kwargs,
    )
    if prior_len:
        dropped = prior_len - len(train_dataset)
        if dropped:
            LOG.warning(
                f"Dropped {dropped} samples with no trainable tokens from train dataset"
            )

    if eval_dataset:
        try:
            prior_len = len(eval_dataset)
        except TypeError:
            # handle iterable datasets case
            prior_len = None
        eval_dataset = eval_dataset.filter(
            drop_no_trainable_tokens,
            **filter_map_kwargs,
            **drop_long_kwargs,
        )
        if prior_len:
            dropped = prior_len - len(eval_dataset)
            if dropped:
                LOG.warning(
                    f"Dropped {dropped} samples with no trainable tokens from eval dataset"
                )

    if cfg.group_by_length:
        train_dataset = train_dataset.map(
            add_length,
            num_proc=cfg.dataset_processes,
            load_from_cache_file=not cfg.is_preprocess,
            desc="Group By Length",
        )

    if cfg.use_pose:
        pose_kwargs = {}
        if cfg.pose_num_chunks is not None:
            pose_kwargs["num_chunks"] = cfg.pose_num_chunks
        if cfg.pose_probability is not None:
            pose_kwargs["pose_probability"] = cfg.pose_probability
        pose_fn = partial(
            add_pose_position_ids,
            max_context_len=cfg.pose_max_context_len,
            split_on_token_ids=cfg.pose_split_on_token_ids,
            **pose_kwargs,
        )
        train_dataset = train_dataset.map(
            pose_fn,
            num_proc=cfg.dataset_processes,
            load_from_cache_file=not cfg.is_preprocess,
            desc="Add position_id column (PoSE)",
        )
        train_dataset = train_dataset.sort("sequence_len")
        if cfg.eval_sample_packing is not False:
            if eval_dataset:
                eval_dataset = eval_dataset.map(
                    pose_fn,
                    num_proc=cfg.dataset_processes,
                    load_from_cache_file=not cfg.is_preprocess,
                    desc="Add position_id column (PoSE)",
                )
    elif cfg.sample_packing:
        drop_long_kwargs = {}
        if filter_map_kwargs:
            drop_long_kwargs["desc"] = "Add position_id column (Sample Packing)"
        train_dataset = train_dataset.map(
            add_position_ids,
            batched=True,
            **filter_map_kwargs,
            **drop_long_kwargs,
        )
        if cfg.eval_sample_packing:
            if eval_dataset:
                eval_dataset = eval_dataset.map(
                    add_position_ids,
                    **filter_map_kwargs,
                    **drop_long_kwargs,
                )

    return train_dataset, eval_dataset


def process_pretraining_datasets_for_packing(
    train_dataset, sequence_len, skip_position_ids=True, drop_attention_mask=False
):
    drop_long = partial(drop_long_seq, sequence_len=sequence_len)

    train_dataset = train_dataset.filter(
        drop_long,
        desc="Dropping Long Sequences",
        load_from_cache_file=False,
    )
    if not skip_position_ids:
        train_dataset = train_dataset.map(
            add_position_ids,
            desc="Add position_id column (Pretraining Sample Packing)",
        )
    if drop_attention_mask:
        train_dataset = train_dataset.remove_columns("attention_mask")

    return train_dataset


def calculate_total_num_steps(cfg, train_dataset, update=True):
    if (
        not cfg.total_num_tokens
        and not cfg.skip_prepare_dataset
        and not cfg.reward_model
    ):
        total_num_tokens = np.sum(
            train_dataset.select_columns("input_ids")
            .to_pandas()["input_ids"]
            .apply(len)
            .values
        )
        LOG.debug(f"total_num_tokens: {total_num_tokens:_}", main_process_only=True)
        if update:
            cfg.total_num_tokens = total_num_tokens

    skip_estimates = cfg.model_config_type == "mamba"

    if (
        not skip_estimates
        and not cfg.total_supervised_tokens
        and not cfg.skip_prepare_dataset
        and not cfg.reward_model
    ):
        total_supervised_tokens = (
            train_dataset.data.column("labels")
            .to_pandas()
            .apply(lambda x: np.sum(np.array(x) != -100))
            .sum()
        )
        LOG.debug(
            f"`total_supervised_tokens: {total_supervised_tokens:_}`",
            main_process_only=True,
        )
        if update:
            cfg.total_supervised_tokens = total_supervised_tokens

    if not skip_estimates and cfg.sample_packing:
        # we have to drop anything longer then sequence len otherwise
        # flash attention with position ids fails

        if cfg.sample_packing_eff_est:
            total_num_steps = (
                # match count to len est in dataloader
                int(
                    math.floor(
                        0.99
                        * cfg.total_num_tokens
                        / cfg.sample_packing_eff_est
                        / cfg.sequence_len
                        // cfg.batch_size
                    )
                    - 1
                )
                * cfg.num_epochs
                * cfg.sequence_parallel_degree
            )
            LOG.debug(
                f"total_num_tokens: {cfg.total_num_tokens:_}, total_num_steps: {total_num_steps:_}",
                main_process_only=True,
            )
        else:
            if cfg.flash_attention and not cfg.multipack_real_batches:
                sampler_batch_size = 1
                batch_max_len = cfg.micro_batch_size * cfg.sequence_len
            else:
                sampler_batch_size = cfg.micro_batch_size
                batch_max_len = cfg.sequence_len
            if cfg.curriculum_sampling:
                sampler = SequentialSampler(train_dataset)
            else:
                sampler = RandomSampler(train_dataset)
            sampler = MultipackBatchSampler(
                sampler=sampler,
                lengths=get_dataset_lengths(train_dataset),
                batch_size=sampler_batch_size,
                batch_max_len=batch_max_len,
                group_size=cfg.sample_packing_group_size,
                bin_size=cfg.sample_packing_bin_size,
                sequential=cfg.sample_packing_sequentially,
                drop_last=True,
            )

            data_loader = DataLoader(
                train_dataset.remove_columns(["length"]),
                batch_sampler=sampler,
            )
            data_loader_len = len(data_loader) * cfg.micro_batch_size // cfg.batch_size
            LOG.debug(f"data_loader_len: {data_loader_len}", main_process_only=True)
            # FIXME: is there a bug here somewhere? the total num steps depends
            # on the agreed on value for sample_packing_eff_est
            total_num_steps = int(
                math.floor(
                    data_loader_len * cfg.num_epochs * cfg.sequence_parallel_degree
                )
            )

            def calc_sample_packing_eff_est(estimates: List[float]):
                LOG.info(f"sample_packing_eff_est across ranks: {repr(estimates)}")
                return max(estimates)

            sample_packing_actual_eff_all = reduce_and_broadcast(
                lambda: sampler.efficiency(),  # pylint: disable=unnecessary-lambda
                calc_sample_packing_eff_est,
            )
            sample_packing_eff_est = (
                math.ceil(sample_packing_actual_eff_all * 100.0) / 100.0
            )
            if update:
                cfg.sample_packing_eff_est = sample_packing_eff_est
            LOG.debug(
                f"sample_packing_eff_est: {cfg.sample_packing_eff_est}",
                main_process_only=True,
            )
    else:
        total_num_steps = int(
            math.ceil(
                len(train_dataset)
                * cfg.num_epochs
                * cfg.sequence_parallel_degree
                / cfg.batch_size
            )
        )
    LOG.debug(f"total_num_steps: {total_num_steps}", main_process_only=True)
    return total_num_steps


def setup_torch_compile_env(cfg):
    if cfg.torch_compile:
        if not cfg.torch_compile_backend:
            os.environ["ACCELERATE_DYNAMO_BACKEND"] = "INDUCTOR"
        else:
            os.environ["ACCELERATE_DYNAMO_BACKEND"] = cfg.torch_compile_backend.upper()


def setup_deepspeed_env(cfg, stage=None):
    from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig

    from axolotl.utils.distributed import distributed_state

    if distributed_state and distributed_state.initialized:
        raise RuntimeError(
            "Distributed State already initialized before Deepspeed setup"
        )

    os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"
    os.environ["ACCELERATE_DEEPSPEED_CONFIG_FILE"] = cfg.deepspeed
    if stage:
        os.environ["ACCELERATE_DEEPSPEED_ZERO_STAGE"] = str(stage)
        if stage == 3:
            os.environ["ACCELERATE_DEEPSPEED_ZERO3_INIT"] = "true"
    # If we don't assign this, it doesn't actually get set in the accelerate weakref
    _ = HfTrainerDeepSpeedConfig(cfg.deepspeed)


def setup_fsdp_envs(cfg):
    os.environ["ACCELERATE_USE_FSDP"] = "true"
    if str(cfg.fsdp_config.fsdp_version) == "2":
        os.environ["FSDP_VERSION"] = "2"
    if cfg.fsdp_config.fsdp_activation_checkpointing:
        os.environ["FSDP_ACTIVATION_CHECKPOINTING"] = "true"
    if cfg.fsdp_config.fsdp_offload_params:
        os.environ["FSDP_OFFLOAD_PARAMS"] = "true"
    if cfg.fsdp_config.fsdp_sync_module_states:
        os.environ["FSDP_SYNC_MODULE_STATES"] = "true"
    if cfg.fsdp_config.fsdp_cpu_ram_efficient_loading:
        os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "true"
    if cfg.fsdp_config.fsdp_use_orig_params:
        os.environ["FSDP_USE_ORIG_PARAMS"] = "true"
    if cfg.fsdp_config.fsdp_state_dict_type:
        os.environ["FSDP_STATE_DICT_TYPE"] = cfg.fsdp_config.fsdp_state_dict_type
    if cfg.fsdp_config.fsdp_auto_wrap_policy:
        os.environ["FSDP_AUTO_WRAP_POLICY"] = cfg.fsdp_config.fsdp_auto_wrap_policy
    if cfg.fsdp_config.fsdp_transformer_layer_cls_to_wrap:
        os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = (
            cfg.fsdp_config.fsdp_transformer_layer_cls_to_wrap
        )
    if cfg.fsdp_config.fsdp_reshard_after_forward is not None:
        os.environ["FSDP_RESHARD_AFTER_FORWARD"] = (
            "true" if cfg.fsdp_config.fsdp_reshard_after_forward else "false"
        )


def prepare_optim_env(cfg):
    if not check_cuda_p2p_ib_support():
        if os.getenv("NCCL_P2P_DISABLE") is None:
            os.environ["NCCL_P2P_DISABLE"] = "1"
    if cfg.fsdp:
        setup_fsdp_envs(cfg)
    elif cfg.deepspeed:
        stage = None
        # check if the cfg.deepspeed is a file
        if os.path.isfile(cfg.deepspeed):
            # parse with json
            with open(cfg.deepspeed, "r", encoding="utf-8") as fin:
                deepspeed_config = json.load(fin)
            stage = deepspeed_config.get("zero_optimization", {}).get("stage", None)
        setup_deepspeed_env(cfg, stage=stage)

    setup_torch_compile_env(cfg)

    if cfg.fp8:
        os.environ["ACCELERATE_MIXED_PRECISION"] = "fp8"
    elif (cfg.bf16 == "auto" and is_torch_bf16_gpu_available()) or cfg.bf16 is True:
        os.environ["ACCELERATE_MIXED_PRECISION"] = "bf16"
    elif cfg.fp16:
        os.environ["ACCELERATE_MIXED_PRECISION"] = "fp16"
    else:
        os.environ["ACCELERATE_MIXED_PRECISION"] = "no"


def prepare_opinionated_env(cfg):
    if cfg.qlora_sharded_model_loading:
        # model loading is forked after the tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_trainer(
    cfg,
    train_dataset,
    eval_dataset,
    model,
    tokenizer,
    processor,
    total_num_steps,
    model_ref=None,
    peft_config=None,
):
    """
    Helper method for instantiating and building a (causal or RLHF) trainer.

    Args:
        cfg: Axolotl config object containing training parameters.
        train_dataset: Dataset to use for training.
        eval_dataset: Dataset to use for evaluation.
        model: The model to train.
        tokenizer: Tokenizer for processing text input.
        processor: Processor for data preparation.
        total_num_steps: The total number of training steps.
        model_ref: Optional reference model for RLHF training. Default is None.
        peft_config: Optional PEFT (Parameter-Efficient Fine-Tuning) configuration. Default is None.

    Returns:
        A trainer instance (either `HFRLTrainer` or `HFCausalTrainer`) configured based
            on the provided parameters.
    """
    if (
        cfg.torch_compile
        and cfg.fsdp_config
        and str(cfg.fsdp_config.fsdp_version) == "2"
    ):
        patch_evaluation_loop_for_fsdp2()
    if cfg.rl:
        trainer_builder = HFRLTrainerBuilder(cfg, model, tokenizer, processor)
        trainer_builder.model_ref = model_ref
        trainer_builder.peft_config = peft_config
    else:
        trainer_builder = HFCausalTrainerBuilder(cfg, model, tokenizer, processor)

    trainer_builder.train_dataset = train_dataset
    trainer_builder.eval_dataset = eval_dataset

    return trainer_builder.build(total_num_steps)
