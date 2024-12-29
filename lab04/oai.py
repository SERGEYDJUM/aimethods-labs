from .llm import AsyncGenericModel, LLMMessage, LLMRole
from openai import AsyncOpenAI


class AsyncGPT4(AsyncGenericModel):
    def __init__(self):
        self.model = "gpt-4o-mini"
        self.client = AsyncOpenAI()

    async def invoke(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        remind_to_end: bool = True,
        **kwargs,
    ) -> str:
        """Answer the prompt asynchronously.

        Args:
            user_prompt (str): User's input text.
            system_prompt (str | None, optional): Instructions for LLM. Defaults to None.
            remind_to_end: (bool, optional): Unused.
        """

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        response = await self.client.chat.completions.create(
            model=self.model, messages=messages, n=1
        )

        if response.choices[0].finish_reason != "stop":
            raise RuntimeError("text processing: unsuccessfull generation")
        elif response.choices[0].message.refusal:
            raise RuntimeError("text processing: model refusal")

        return response.choices[0].message.content

    async def invoke_messages(
        self,
        messages: list[LLMMessage],
        remind_to_end: bool = True,
        max_new_tokens: int = 64,
        **kwargs,
    ) -> str:
        """Generates text for next message in chat.

        Args:
            messages (list[LLMMessage]): Message history.
            remind_to_end (bool, optional): Unused.
            max_new_tokens (int, optional): Unused.
        """
        if not messages:
            raise ValueError("No messages passed")

        if messages[-1].role == LLMRole.ASSISTANT:
            raise ValueError("Last message must be from user or system")

        out_messages = []

        for msg in messages:
            if msg.role == LLMRole.USER:
                out_messages.append(
                    {
                        "role": "user",
                        "content": msg.content,
                    },
                )
            elif msg.role == LLMRole.ASSISTANT:
                out_messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content,
                    },
                )
            else:
                out_messages.append(
                    {
                        "role": "system",
                        "content": msg.content,
                    },
                )

        response = await self.client.chat.completions.create(
            model=self.model, messages=out_messages, n=1
        )

        if response.choices[0].finish_reason != "stop":
            raise RuntimeError("text processing: unsuccessfull generation")
        elif response.choices[0].message.refusal:
            raise RuntimeError("text processing: model refusal")

        return response.choices[0].message.content
