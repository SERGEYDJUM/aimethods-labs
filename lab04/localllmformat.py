from exllamav2 import ExLlamaV2Tokenizer
from torch import Tensor


class Llama3PromptFormat:
    """Llama-3's default prompt related configuraion
    and convenience functions.
    """

    botname = "Chatbot"
    username = "User"
    description = "Llama-3-instruct models"

    def __init__(self):
        pass

    def default_system_prompt(self) -> str:
        return (
            """Assist users with tasks and answer questions to the best of your knowledge. Provide helpful and informative """
            + """responses. Be conversational and engaging. If you are unsure or lack knowledge on a topic, admit it and try """
            + """to find the answer or suggest where to find it. Keep responses concise and relevant. Follow ethical """
            + """guidelines and promote a safe and respectful interaction."""
        )

    def user_prompt(self, raw_prompt: str) -> str:
        template = (
            "<|start_header_id|>user<|end_header_id|>\n\n" "{raw_prompt}<|eot_id|>"
        )
        return template.format(raw_prompt=raw_prompt)

    def system_prompt(self, raw_prompt: str) -> str:
        template = (
            "<|start_header_id|>system<|end_header_id|>\n\n" "{raw_prompt}<|eot_id|>"
        )
        return template.format(raw_prompt=raw_prompt)

    def assistant_prompt(self, raw_prompt: str) -> str:
        template = (
            "<|start_header_id|>assistant<|end_header_id|>\n\n" "{raw_prompt}<|eot_id|>"
        )
        return template.format(raw_prompt=raw_prompt)

    def assistant_invitation(self) -> str:
        return "<|start_header_id|>assistant<|end_header_id|>\n\n"

    def end_reminder(self) -> str:
        stopper = self._stop_conditions_strings()[0]
        return f'\nDo not forget to add "{stopper}" after your response.'

    def _stop_conditions_strings(self) -> list[str]:
        return ["<|eot_id|>", "<|start_header_id|>"]

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
        """Returns: add_bos, add_eos, encode_special_tokens"""
        return True, False, True

    def format_prompt(
        self, user_prompt: str, system_prompt: str | None = None, first: bool = True
    ) -> str:
        """Format prompts specifically for Llama-3 models."""

        prompt = ""

        if not system_prompt:
            system_prompt = self.default_system_prompt()

        if first:
            prompt += self.system_prompt(system_prompt)

        prompt += self.user_prompt(user_prompt)
        prompt += self.assistant_invitation()
        return prompt

    def encode_prompt(self, tokenizer: ExLlamaV2Tokenizer, text: str) -> Tensor:
        """Encode final prompt text in a Llama-3 specific way."""

        add_bos, add_eos, encode_special_tokens = self._encoding_options()
        return tokenizer.encode(
            text,
            add_bos=add_bos,
            add_eos=add_eos,
            encode_special_tokens=encode_special_tokens,
        )
