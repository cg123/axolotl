"""Module testing prompters"""

import unittest

from rathe import AlpacaPromptFormatter, InstructPrompt


class AlpacaPrompterTest(unittest.TestCase):
    """
    Test AlpacaPrompter
    """

    def test_prompt_style_w_none(self):
        formatter = AlpacaPromptFormatter()
        res = formatter.format(
            InstructPrompt("tell me a joke"), special_tokens={}
        ).to_string()
        # just testing that it uses instruct style
        assert "### Instruction:" in res

    def test_prompt_style_w_instruct(self):
        formatter = AlpacaPromptFormatter()
        res = formatter.format(
            InstructPrompt("tell me a joke about the following", input="alpacas"),
            special_tokens={},
        ).to_string()
        assert "Below is an instruction" in res
        assert "### Instruction:" in res
        assert "### Input:" in res
        assert "alpacas" in res
        assert "### Response:" in res
        assert "USER:" not in res
        assert "ASSISTANT:" not in res
        res = formatter.format(
            InstructPrompt("tell me a joke about the following"), special_tokens={}
        ).to_string()
        assert "Below is an instruction" in res
        assert "### Instruction:" in res
        assert "### Input:" not in res
        assert "### Response:" in res
        assert "USER:" not in res
        assert "ASSISTANT:" not in res

    def test_prompt_style_w_chat(self):
        formatter = AlpacaPromptFormatter()
        res = formatter.format(
            InstructPrompt(
                "tell me a joke about the following", input="alpacas"
            ).as_chat(),
            special_tokens={},
        ).to_string()
        assert "Below is an instruction" in res
        assert "### Instruction:" not in res
        assert "### Input:" not in res
        assert "alpacas" in res
        assert "### Response:" not in res
        assert "USER:" in res
        assert "ASSISTANT:" in res
        res = formatter.format(
            InstructPrompt("tell me a joke about the following").as_chat(),
            special_tokens={},
        ).to_string()
        assert "Below is an instruction" in res
        assert "### Instruction:" not in res
        assert "### Input:" not in res
        assert "### Response:" not in res
        assert "USER:" in res
        assert "ASSISTANT:" in res

    def test_system_prompt(self):
        prompter = SystemDataPrompter(prompt_style=PromptStyle.CHAT.value)
        res = next(
            prompter.build_prompt_w_system(
                "use cot", "tell me a joke about the following", "alpacas"
            )
        )
        assert "use cot" in res
        assert res.startswith("SYSTEM:")
        assert "### Instruction:" not in res
        assert "### Input:" not in res
        assert "alpacas" in res
        assert "### Response:" not in res
        assert "USER:" in res
        assert "ASSISTANT:" in res


class UnpromptedPrompterTest(unittest.TestCase):
    """
    Test class for UnpromptedPrompter with no system prompts
    """

    def test_prompt_style_w_none(self):
        prompter = UnpromptedPrompter(prompt_style=None)
        res = next(prompter.build_prompt("tell me a joke"))
        assert "### Instruction:" in res
        assert "tell me a joke" in res
        assert res.startswith("###")

    def test_prompt_style_w_instruct(self):
        prompter = UnpromptedPrompter(prompt_style=PromptStyle.INSTRUCT.value)
        res = next(
            prompter.build_prompt("tell me a joke about the following", "alpacas")
        )
        assert "### Instruction:" in res
        assert "tell me a joke" in res
        assert res.startswith("###")

    def test_prompt_style_w_chat(self):
        prompter = UnpromptedPrompter(prompt_style=PromptStyle.CHAT.value)
        res = next(
            prompter.build_prompt("tell me a joke about the following", "alpacas")
        )
        assert "USER:" in res
        assert "tell me a joke" in res
        assert res.startswith("USER:")


class MultipleChoiceExplainPrompterTest(unittest.TestCase):
    """
    Test class for MultipleChoiceExplainPrompter
    """

    def test_prompt_style_w_chat(self):
        prompter = MultipleChoiceExplainPrompter(prompt_style=PromptStyle.CHAT.value)
        res = next(prompter.build_prompt("choose one", "- A\n- B\n- C", "C"))
        assert "USER:" in res
        assert "choose one" in res
        assert "Choose the answer that best answers the question." in res
        assert "- A\n- B\n- C" in res
