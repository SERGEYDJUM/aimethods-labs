import os, asyncio
from loguru import logger
from uuid import UUID, uuid4
from exllamav2.generator import ExLlamaV2DynamicGeneratorAsync, ExLlamaV2DynamicJobAsync
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
    ExLlamaV2Cache,
)


class AsyncLLM:
    def __init__(self, model_dir: str = os.environ["EXL_MODEL_DIR"]):
        config = ExLlamaV2Config(model_dir)
        config.arch_compat_overrides()

        model = ExLlamaV2(config)
        cache = ExLlamaV2Cache(model, lazy=True, max_seq_len=2048)
        model.load_autosplit(cache, progress=True)

        self.tokenizer = ExLlamaV2Tokenizer(config)

        self.generator = ExLlamaV2DynamicGeneratorAsync(
            model=model,
            cache=cache,
            tokenizer=self.tokenizer,
            paged=False, #Disable Linux-only Flash Attention 2, removes job batch processing
        )

    async def complete_text(self, prompt: str, job_id: UUID):
        # Create an iterable, async job. The job will be transparently batched 
        # together with other jobs on the generator if possible.
        job = ExLlamaV2DynamicJobAsync(
            self.generator,
            input_ids=self.tokenizer.encode(prompt, add_bos=False),
            max_new_tokens=100,
        )

        text = prompt

        async for new_text in job:
            text += new_text.get("text", "")
            
        logger.debug(f"llm: completed job {job_id}")

        return text


if __name__ == "__main__":
    async def test():
        llm = AsyncLLM()

        prompts = [
            "Once upon a time, there was",
            "asyncio in Python is a great feature because",
        ]

        tasks = [llm.complete_text(prompt, uuid4()) for prompt in prompts]
        outputs = await asyncio.gather(*tasks)

        print()
        for i, output in enumerate(outputs):
            print(f"Output {i}")
            print("-----------")
            print(output)
            print()

        await llm.generator.close()

    asyncio.run(test())
