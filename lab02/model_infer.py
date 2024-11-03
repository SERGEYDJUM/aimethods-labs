from transformers import GPT2LMHeadModel, GPT2TokenizerFast


class Model:
    def __init__(self) -> None:
        self.tokenizer = GPT2TokenizerFast.from_pretrained(
            "ai-forever/rugpt3large_based_on_gpt2"
        )
        self.model = GPT2LMHeadModel.from_pretrained(
            "ai-forever/rugpt3large_based_on_gpt2",
            torch_dtype="float16",
            device_map="cuda:0",
        )

    def execute_prompt(self, prompt: str, **kwargs) -> str:
        tokens = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        gen_tokens = self.model.generate(
            tokens.input_ids,
            do_sample=True,
            **kwargs,
        )

        return self.tokenizer.batch_decode(gen_tokens)[0]
