import os, asyncio
from exllamav2.generator import ExLlamaV2DynamicGeneratorAsync, ExLlamaV2DynamicJobAsync
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Tokenizer,
    ExLlamaV2Cache,
)

from prompt import Phi3PromptFormat


class AsyncPhi3:
    """Async inference for Phi-3 family models."""

    def __init__(self, model_dir: str = os.environ["EXL_MODEL_DIR"]) -> None:
        """Loads a model from Phi-3 family and it's tokenizer.

        Args:
            model_dir (str, optional): Directory containing weights and config. Defaults to os.environ["EXL_MODEL_DIR"].
        """
        config = ExLlamaV2Config(model_dir)
        model = ExLlamaV2(config)
        cache = ExLlamaV2Cache(model, max_seq_len=2048)
        model.load_autosplit(cache, progress=True)

        self.tokenizer = ExLlamaV2Tokenizer(config)
        self.prompt_format = Phi3PromptFormat()
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

    async def complete_text(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        max_new_tokens: int = 64,
        **kwargs,
    ) -> str:
        """Anwer the prompt asynchronously.

        Args:
            user_prompt (str): User's input text.
            system_prompt (str | None, optional): Instructions for LLM. Defaults to None.
            max_new_tokens (int, optional): Maximum length of the output. Defaults to 64.
        """

        # Create an iterable, async job. The job will be transparently batched
        # together with other jobs on the generator if possible.
        prompt = self.prompt_format.format_prompt(
            user_prompt=user_prompt, system_prompt=system_prompt
        )

        job = ExLlamaV2DynamicJobAsync(
            self.generator,
            input_ids=self.prompt_format.encode_prompt(self.tokenizer, prompt),
            max_new_tokens=max_new_tokens,
            stop_conditions=self.stop_conditions,
            **kwargs,
        )

        return "".join([frag.get("text", "") async for frag in job])


if __name__ == "__main__":

    async def test():
        llm = AsyncPhi3()

        prompts = [
            "Меня зовут Сергей",
            "Николай дома? Я Альберт, его друг.",
        ]

        tasks = [
            llm.complete_text(
                prompt,
                system_prompt=(
                    "You are tasked with extracting a name from the input text. "
                    "If text contains more than one name, pick one that is most likely user's. "
                    'You must end your response after that with "<|end|>".'
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
