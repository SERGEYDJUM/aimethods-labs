import asyncio
from dataclasses import dataclass
from typing import Any
from loguru import logger
from os import getenv

from aiogram import Bot, Dispatcher, Router, html
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext
from datetime import datetime
from .llm import AsyncGenericModel, AsyncLlama3, LLMMessage, LLMRole
from .oai import AsyncGPT4

from .intents import (
    Agreement,
    extract_cosmetic_care,
    extract_number,
    extract_name,
    extract_agreement,
    extract_serious_care,
)

from .utils import (
    BOT_ANSWER_SYS,
    BOT_GLOBAL_SYS,
    GENERIC_RESPONSE_REPEAT,
    PROTOCOL,
    problem_resolve,
    Specialist,
    CareType,
)


@dataclass
class ModelProvider:
    local_model: AsyncGenericModel = AsyncGenericModel()
    oai_model: AsyncGenericModel = AsyncGenericModel()


class states(StatesGroup):
    """States the bot can be in for each user"""

    start = State()
    name_extraction = State()
    care_category = State()
    serious_care = State()
    cosmetic_care = State()
    care_confirmation = State()
    date_confirmation = State()
    number_extraction = State()
    final_confirmation = State()
    end = State()


router = Router()
"""Bot's main router"""

models: ModelProvider = ModelProvider()


async def bot_answer(
    message: Message,
    state: FSMContext,
    example: str,
    purpose: str = "Not given, deduce from text.",
    disable_ai: bool = False,
) -> None:
    """Creates AI generated response from example ans purpose.

    Args:
        example (str): Generic reply example.
        purpose (str, optional): Purpose of the reply. Defaults to "Not given, deduce from text.".
        disable_ai (bool, optional): Passes example to output with no AI. Defaults to False.
    """
    data = await state.get_data()
    history, local = data["history"], data["local"]

    llm_messages = [BOT_GLOBAL_SYS]
    llm_messages.extend(history)
    llm_messages.append(
        LLMMessage(
            LLMRole.SYSTEM, BOT_ANSWER_SYS.format(example=example, purpose=purpose)
        )
    )

    resp = example

    if not disable_ai:
        resp = await (
            models.local_model if local else models.oai_model
        ).invoke_messages(llm_messages, max_new_tokens=256)

    history.append(LLMMessage(LLMRole.ASSISTANT, resp))
    await state.update_data(history=history)
    await message.answer(resp)


async def store_user_message(message: Message, state: FSMContext) -> dict[str, Any]:
    """Saves user message to history.

    Returns:
        dict[str, Any]: Updated context data.
    """
    data = await state.get_data()
    data["history"].append(LLMMessage(LLMRole.USER, message.text))
    return await state.update_data(history=data["history"])


@router.message(CommandStart())
async def command_start_handler(message: Message, state: FSMContext) -> None:
    """None -> Initial -> Name Extraction"""
    await state.set_state(states.start)
    logger.debug(f"{message.from_user.full_name} in {await state.get_state()}")

    await state.set_data(
        {
            "local": "gpt" not in message.text,
            "history": [
                LLMMessage(LLMRole.USER, "/start"),
                LLMMessage(
                    LLMRole.ASSISTANT,
                    "Hello, this is AIcare clinic. What is your name?",
                ),
            ],
        }
    )

    await message.answer("Hello, this is AIcare clinic. What is your name?")
    await state.set_state(states.name_extraction)


@router.message(states.name_extraction)
async def resolve_name(message: Message, state: FSMContext) -> None:
    """Name Extraction -> Care Category Extraction | Self"""
    logger.debug(f"{message.from_user.full_name} in {await state.get_state()}")

    data = await store_user_message(message, state)

    if name := await extract_name(
        message.text, models.local_model if data["local"] else models.oai_model
    ):
        await state.update_data(user_name=name)
        await bot_answer(
            message,
            state,
            f"{name}, are you here because of a dental health {html.bold('problem')}?",
            "Determine whether user has a real and urgent problem or not.",
        )
        await state.set_state(states.care_category)
    else:
        await bot_answer(
            message,
            state,
            "Unfortunately, I didn't understand. Can you tell me your first name?",
            "Determine user's first name.",
        )


@router.message(states.care_category)
async def resolve_care_cat(message: Message, state: FSMContext) -> None:
    """Category Extraction -> [Serious Care, Cosmetic Care, Care Confirmation (for examination)] | Self"""
    logger.debug(f"{message.from_user.full_name} in {await state.get_state()}")

    data = await store_user_message(message, state)

    if ag_level := await extract_agreement(
        message.text, models.local_model if data["local"] else models.oai_model
    ):
        if ag_level == Agreement.YES:
            await bot_answer(
                message,
                state,
                "What is the problem exactly?",
                "Determine the type of dental health issue.",
            )
            await state.set_state(states.serious_care)
        elif ag_level == Agreement.NO:
            user_name = (await state.get_data())["user_name"]
            await bot_answer(
                message,
                state,
                f"AIcare provides a few cosmetic services. How can I help you, {user_name}?",
                "Determine the kind of cosmetic service.",
            )
            await state.set_state(states.cosmetic_care)
        else:
            await state.update_data(
                specialist=Specialist.THERAPIST,
                care_type=CareType.EXAMINATION,
            )
            await bot_answer(
                message,
                state,
                "If you are not sure, I can make an appointment for you to see a dental therapist, who will determine your problem. Ok?",
                "Get user's agreement or disagreement.",
            )
            await state.set_state(states.care_confirmation)
    else:
        await bot_answer(message, state, GENERIC_RESPONSE_REPEAT)


@router.message(states.serious_care)
async def resolve_serious_care(message: Message, state: FSMContext) -> None:
    """Serious Care -> Care Confirmation (serious) | Self"""
    logger.debug(f"{message.from_user.full_name} in {await state.get_state()}")

    data = await store_user_message(message, state)

    if sc_type := await extract_serious_care(
        message.text, models.local_model if data["local"] else models.oai_model
    ):
        specialist, care, answer = problem_resolve(sc_type)
        await state.update_data(specialist=specialist, care_type=care)
        await bot_answer(message, state, answer)
        await state.set_state(states.care_confirmation)
    else:
        await bot_answer(message, state, GENERIC_RESPONSE_REPEAT)


@router.message(states.cosmetic_care)
async def resolve_cosmetic_care(message: Message, state: FSMContext) -> None:
    """Cosmetic Care -> Care Confirmation (cosmetic) | Self"""
    logger.debug(f"{message.from_user.full_name} in {await state.get_state()}")

    data = await store_user_message(message, state)

    if sc_type := await extract_cosmetic_care(
        message.text, models.local_model if data["local"] else models.oai_model
    ):
        specialist, care, answer = problem_resolve(sc_type)
        await state.update_data(specialist=specialist, care_type=care)
        await bot_answer(message, state, answer)
        await state.set_state(states.care_confirmation)
    else:
        await bot_answer(message, state, GENERIC_RESPONSE_REPEAT)


@router.message(states.care_confirmation)
async def resolve_care_confirmation(message: Message, state: FSMContext) -> None:
    """Care Confirmation (serious, cosmetic) -> [Date Confirmation, Category Extraction] | Self"""
    logger.debug(f"{message.from_user.full_name} in {await state.get_state()}")

    data = await store_user_message(message, state)

    if ag_level := await extract_agreement(
        message.text, models.local_model if data["local"] else models.oai_model
    ):
        if ag_level == Agreement.YES:
            date = datetime.now()
            await state.update_data(date=date)
            await bot_answer(
                message,
                state,
                f"Are you fine with the following date and time: {date.isoformat(timespec='minutes')}?",
                "To confirm that appointment time fits the user.",
            )
            await state.set_state(states.date_confirmation)
        else:
            await bot_answer(
                message,
                state,
                "Ok. Then let's start from the beginning. Are you here because of a serious dental health problem?",
                "Determine whether user has an urgent, non-cosmetic problem.",
            )
            await state.set_state(states.care_category)
    else:
        await bot_answer(message, state, GENERIC_RESPONSE_REPEAT)


@router.message(states.date_confirmation)
async def resolve_date_confirmation(message: Message, state: FSMContext) -> None:
    """Date Confirmation -> [Number Extraction, Self, End] | Self"""
    logger.debug(f"{message.from_user.full_name} in {await state.get_state()}")

    data = await store_user_message(message, state)

    if ag_level := await extract_agreement(
        message.text, models.local_model if data["local"] else models.oai_model
    ):
        if ag_level == Agreement.YES:
            await bot_answer(
                message,
                state,
                "We need your phone number to remind you about an appointment.",
                "Determine user's phone number.",
            )
            await state.set_state(states.number_extraction)
        elif ag_level == Agreement.UNSURE:
            await bot_answer(
                message,
                state,
                "Sorry, this is the only available time. Should I make an appointment, after all?",
                "Determine user's agreement or disagreement.",
            )
        else:
            await bot_answer(
                message,
                state,
                "We cannot provide you a different time slot. Sorry and goodbye.",
            )
            await state.set_state(states.end)
    else:
        await bot_answer(message, state, GENERIC_RESPONSE_REPEAT)


@router.message(states.number_extraction)
async def resolve_number(message: Message, state: FSMContext) -> None:
    """Number Extraction -> Final Confirmation | Self"""
    logger.debug(f"{message.from_user.full_name} in {await state.get_state()}")

    data = await store_user_message(message, state)

    if phone := await extract_number(
        message.text, models.local_model if data["local"] else models.oai_model
    ):
        data = await state.update_data(phone=phone)
        await bot_answer(
            message,
            state,
            PROTOCOL.format(**data),
            "Present final appointment information to user and ask for user's agreement.",
        )
        await state.set_state(states.final_confirmation)
    else:
        await bot_answer(message, state, GENERIC_RESPONSE_REPEAT)


@router.message(states.final_confirmation)
async def resolve_final_confirmation(message: Message, state: FSMContext) -> None:
    """Final Confirmation -> [End, Self, End] | Self"""
    logger.debug(f"{message.from_user.full_name} in {await state.get_state()}")

    data = await store_user_message(message, state)

    if ag_level := await extract_agreement(
        message.text, models.local_model if data["local"] else models.oai_model
    ):
        if ag_level == Agreement.YES:
            await bot_answer(message, state, "We will be waiting for you. Goodbye.")
            await state.set_state(states.end)
        elif ag_level == Agreement.UNSURE:
            await bot_answer(
                message,
                state,
                "You must decide now. Should we make this appointment anyway?",
                "Trying to get aggrement for appointment from user again.",
            )
        else:
            await bot_answer(message, state, "Goodbye then.")
            await state.set_state(states.end)
    else:
        await bot_answer(message, state, GENERIC_RESPONSE_REPEAT)


async def main() -> None:
    global models
    # Load models into global scope
    models = ModelProvider(local_model=AsyncLlama3(), oai_model=AsyncGPT4())

    bot = Bot(
        token=getenv("BOT_TOKEN"),
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )

    dp = Dispatcher()
    dp.include_router(router)

    logger.info("Created aiogram bot. Polling...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
