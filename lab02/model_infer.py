from transformers import GPT2LMHeadModel, GPT2TokenizerFast


class Model:
    """'ruGPT-3 Large' model executor"""

    hf_repo = "ai-forever/rugpt3large_based_on_gpt2"

    def __init__(self) -> None:
        """Constructs tokenizer and GPT-2 model.
        Loads checkpoint and configs from HuggingFace repository.
        """

        self.tokenizer = GPT2TokenizerFast.from_pretrained(self.hf_repo)
        self.model = GPT2LMHeadModel.from_pretrained(
            self.hf_repo,
            # Following options assume modern Nvidia GPU
            torch_dtype="float16",
            device_map="cuda:0",
        )

    def execute_prompt(self, prompt: str, **kwargs) -> str:
        """Generates tokens from prompt with pass-through keyword arguments"""

        tokens = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        beam_output = self.model.generate(
            tokens.input_ids,
            # Avoids repeting n-grams of length 2
            no_repeat_ngram_size=2,
            # These two parameters enable Beam-search Multinomial Sampling
            do_sample=True,
            num_beams=4,
            **kwargs,
        )

        return self.tokenizer.decode(beam_output[0], skip_special_tokens=True)
