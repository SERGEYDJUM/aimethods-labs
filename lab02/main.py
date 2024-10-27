from transformers import GPT2LMHeadModel, GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("ai-forever/rugpt3large_based_on_gpt2")
model = GPT2LMHeadModel.from_pretrained(
    "ai-forever/rugpt3large_based_on_gpt2", torch_dtype="float16", device_map="cuda:0"
)

def execute_prompt(prompt: str) -> str:
    tokens = tokenizer(prompt, return_tensors="pt").to(model.device)

    gen_tokens = model.generate(
        tokens.input_ids,
        do_sample=True,
        temperature=1,
        max_new_tokens=128,
    )

    return tokenizer.batch_decode(gen_tokens)[0]


if __name__ == "__main__":
    prompt = """Контрольная научно-исследовательская работа.
    Тема: Исследование многоканальных моделей наблюдателей в медицинской томографии.
    Аннотация:
    """

    print(execute_prompt(prompt))
