from exllamav2 import ExLlamaV2Tokenizer
from torch import Tensor


class Phi3PromptFormat:
    """Phi-3's default prompt related configuraion
    and convenience functions.
    """

    botname = "Chatbot"
    username = "User"
    description = "Phi3-instruct models"

    def __init__(self):
        pass

    def default_system_prompt(self) -> str:
        return "You are a helpful AI assistant."

    def _first_prompt(self, sysprompt: str | None = None) -> str:
        r = """<s>"""
        if sysprompt:
            r += """<|system|>\n""" + """<|system_prompt|>""" + """<|end|>\n"""
        r += """<|user|>\n""" + """<|user_prompt|><|end|>\n""" + """<|assistant|>\n"""
        return r

    def _subs_prompt(self) -> str:
        return (
            """<|end|>\n"""
            + """<|user|>\n"""
            + """<|user_prompt|>"""
            + """<|end|>\n"""
            + """<|assistant|>\n"""
        )

    def _stop_conditions_strings(self) -> list[str]:
        return ["<|end|>", "<|assistant|>", "<|endoftext|>"]

    def stop_conditions_ids(self, tokenizer: ExLlamaV2Tokenizer) -> list[int]:
        stops = [tokenizer.eos_token_id]
        stops.extend(
            filter(
                lambda x: x is not None,
                map(lambda x: tokenizer.single_id(x), self._stop_conditions_strings()),
            )
        )

        return stops

    def _encoding_options(self) -> tuple[bool, bool, bool]:
        return False, False, True

    def _print_extra_newline(self) -> bool:
        return True

    def format_prompt(
        self, user_prompt: str, system_prompt: str | None = None, first: bool = True
    ) -> str:
        """Format prompts specifically for Phi-3 models."""

        if not system_prompt:
            system_prompt = self.default_system_prompt()

        if first:
            return (
                self._first_prompt(system_prompt)
                .replace("<|system_prompt|>", system_prompt)
                .replace("<|user_prompt|>", user_prompt)
            )
        else:
            return self._subs_prompt().replace("<|user_prompt|>", user_prompt)

    def encode_prompt(self, tokenizer: ExLlamaV2Tokenizer, text: str) -> Tensor:
        """Encode final prompt text in a Phi-3 specific way."""

        add_bos, add_eos, encode_special_tokens = self._encoding_options()
        return tokenizer.encode(
            text,
            add_bos=add_bos,
            add_eos=add_eos,
            encode_special_tokens=encode_special_tokens,
        )
