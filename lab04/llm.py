import os, asyncio
from dataclasses import dataclass
from loguru import logger
from enum import Enum
from exllamav2.generator import ExLlamaV2DynamicGeneratorAsync, ExLlamaV2DynamicJobAsync
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Tokenizer,
    ExLlamaV2Cache,
)

from .localllmformat import Llama3PromptFormat


class LLMRole(Enum):
    SYSTEM = 0
    USER = 1
    ASSISTANT = 2


@dataclass
class LLMMessage:
    role: LLMRole
    content: str


class AsyncGenericModel:
    async def invoke(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        remind_to_end: bool = True,
        **kwargs,
    ) -> str:
        raise RuntimeError("Abstract model called")

    async def invoke_messages(
        self, messages: list[LLMMessage], remind_to_end: bool = True, **kwargs
    ) -> str:
        raise RuntimeError("Abstract model called")


class AsyncLlama3(AsyncGenericModel):
    """Async inference for Llama-3 family models."""

    def __init__(self, model_dir: str = os.environ["EXL_MODEL_DIR"]) -> None:
        """Loads a model from Llama-3 family and it's tokenizer.

        Args:
            model_dir (str, optional): Directory containing weights and config. Defaults to EXL_MODEL_DIR from environment.
        """
        config = ExLlamaV2Config(model_dir)
        model = ExLlamaV2(config)
        cache = ExLlamaV2Cache(model, max_seq_len=2048)
        model.load_autosplit(cache, progress=True)

        self.tokenizer = ExLlamaV2Tokenizer(config)
        self.prompt_format = Llama3PromptFormat()

        # These are needed to allow model to stop itself
        self.stop_conditions = list(
            filter(
                lambda x: x is not None,
                self.prompt_format.stop_conditions_ids(self.tokenizer),
            )
        )

        self.generator = ExLlamaV2DynamicGeneratorAsync(
            model=model,
            cache=cache,
            tokenizer=self.tokenizer,
            paged=False,  # Disable Linux-only Flash Attention 2, results in no batching
        )

    async def invoke(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        remind_to_end: bool = True,
        max_new_tokens: int = 64,
        **kwargs,
    ) -> str:
        """Answer the prompt asynchronously.

        Args:
            user_prompt (str): User's input text.
            system_prompt (str | None, optional): Instructions for LLM. Defaults to None.
            remind_to_end: (bool, optional): Add stopping instruction to system_prompt. Defaults to True.
            max_new_tokens (int, optional): Maximum length of the output. Defaults to 64.
        """

        if remind_to_end:
            # For some reason model is unwilling to stop without direct instruction
            system_prompt += self.prompt_format.end_reminder()

        prompt = self.prompt_format.format_prompt(
            user_prompt=user_prompt, system_prompt=system_prompt
        )

        # Create an iterable, async job. The job will be transparently batched
        # together with other jobs on the generator if possible.
        job = ExLlamaV2DynamicJobAsync(
            self.generator,
            input_ids=self.prompt_format.encode_prompt(self.tokenizer, prompt),
            stop_conditions=self.stop_conditions,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

        text = "".join([frag.get("text", "") async for frag in job])
        return text.strip()

    async def invoke_messages(
        self,
        messages: list[LLMMessage],
        remind_to_end: bool = True,
        max_new_tokens: int = 64,
        **kwargs,
    ) -> str:
        """Generates text for next message in chat.

        Args:
            messages (list[LLMMessage]): История сообщений.
            remind_to_end (bool, optional): Add stopping instruction to ALL system prompts. Defaults to True.
            max_new_tokens (int, optional): Maximum length of the output. Defaults to 64.
        """
        if not messages:
            raise ValueError("No messages passed")

        if messages[-1].role == LLMRole.ASSISTANT:
            raise ValueError("Last message must be from user or system")

        prompt = ""

        for msg in messages:
            if msg.role == LLMRole.USER:
                prompt += self.prompt_format.user_prompt(msg.content)
            elif msg.role == LLMRole.ASSISTANT:
                prompt += self.prompt_format.assistant_prompt(msg.content)
            else:
                content = msg.content
                if remind_to_end:
                    content += self.prompt_format.end_reminder()
                prompt += self.prompt_format.system_prompt(content)

        prompt += self.prompt_format.assistant_invitation()

        job = ExLlamaV2DynamicJobAsync(
            self.generator,
            input_ids=self.prompt_format.encode_prompt(self.tokenizer, prompt),
            stop_conditions=self.stop_conditions,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

        text = "".join([frag.get("text", "") async for frag in job])
        return text.strip()


if __name__ == "__main__":

    async def test():
        llm = AsyncLlama3()

        prompts = [
            "Меня зовут Сергей",
            "Николай дома? Я Альберт, его друг.",
        ]

        tasks = [
            llm.invoke(
                prompt,
                system_prompt=(
                    "You are tasked with extracting a name from the input text. "
                    "If text contains more than one name, pick one that is most likely user's. "
                    "If you cannot find a name, respond with nothing."
                ),
            )
            for prompt in prompts
        ]
        outputs = await asyncio.gather(*tasks)

        print()
        for i, output in enumerate(outputs):
            print(f"Output {i}")
            print("-----------")
            print(output)
            print()

        await llm.generator.close()

    asyncio.run(test())
