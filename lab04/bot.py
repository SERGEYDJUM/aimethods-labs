import asyncio
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

from .utils import (
    GENERIC_RESPONSE_REPEAT,
    PROTOCOL,
    problem_resolve,
    Specialist,
    CareType,
)
from .intents import (
    Agreement,
    extract_cosmetic_care,
    extract_number,
    init_intents,
    extract_name,
    extract_agreement,
    extract_serious_care,
)


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


@router.message(CommandStart())
async def command_start_handler(message: Message, state: FSMContext) -> None:
    """None -> Initial -> Name Extraction"""
    await state.set_state(states.start)
    logger.debug(f"{message.from_user.full_name} in {await state.get_state()}")

    await message.answer(
        "Здравствуйте! Это сервис для записи на приём в стоматологию сети AIcare. Как к вам обращаться?"
    )
    await state.set_state(states.name_extraction)


@router.message(states.name_extraction)
async def resolve_name(message: Message, state: FSMContext) -> None:
    """Name Extraction -> Care Category Extraction | Self"""
    logger.debug(f"{message.from_user.full_name} in {await state.get_state()}")

    if name := await extract_name(message.text):
        await state.update_data(user_name=name)
        await message.answer(
            f"{name}, вы обратились к нам из-за {html.bold('проблемы')} со здоровьем полости рта?"
        )
        await state.set_state(states.care_category)
    else:
        await message.answer(
            "К сожалению, я не понимаю. Назовите своё имя, пожалуйста."
        )


@router.message(states.care_category)
async def resolve_care_cat(message: Message, state: FSMContext) -> None:
    """Category Extraction -> [Serious Care, Cosmetic Care, Care Confirmation (for examination)] | Self"""
    logger.debug(f"{message.from_user.full_name} in {await state.get_state()}")

    if ag_level := await extract_agreement(message.text):
        if ag_level == Agreement.YES:
            await message.answer("Что конкретно вас беспокоит?")
            await state.set_state(states.serious_care)
        elif ag_level == Agreement.NO:
            user_name = (await state.get_data())["user_name"]
            await message.answer(
                f"Сеть клиник AIcare предоставляет множество косметических услуг. Чем мы можем помочь вам, {user_name}?"
            )
            await state.set_state(states.cosmetic_care)
        else:
            await state.update_data(
                specialist=Specialist.THERAPIST,
                care_type=CareType.EXAMINATION,
            )
            await message.answer(
                "Если вы не уверены, я запишу вас к стоматологу-терапевту, который определит проблему и направит вас к нужному специалисту. Хорошо?"
            )
            await state.set_state(states.care_confirmation)
    else:
        await message.answer(GENERIC_RESPONSE_REPEAT)


@router.message(states.serious_care)
async def resolve_serious_care(message: Message, state: FSMContext) -> None:
    """Serious Care -> Care Confirmation (serious) | Self"""
    logger.debug(f"{message.from_user.full_name} in {await state.get_state()}")

    if sc_type := await extract_serious_care(message.text):
        specialist, care, answer = problem_resolve(sc_type)
        await state.update_data(specialist=specialist, care_type=care)
        await message.answer(answer)
        await state.set_state(states.care_confirmation)
    else:
        await message.answer(GENERIC_RESPONSE_REPEAT)


@router.message(states.cosmetic_care)
async def resolve_cosmetic_care(message: Message, state: FSMContext) -> None:
    """Cosmetic Care -> Care Confirmation (cosmetic) | Self"""
    logger.debug(f"{message.from_user.full_name} in {await state.get_state()}")

    if sc_type := await extract_cosmetic_care(message.text):
        specialist, care, answer = problem_resolve(sc_type)
        await state.update_data(specialist=specialist, care_type=care)
        await message.answer(answer)
        await state.set_state(states.care_confirmation)
    else:
        await message.answer(GENERIC_RESPONSE_REPEAT)


@router.message(states.care_confirmation)
async def resolve_care_confirmation(message: Message, state: FSMContext) -> None:
    """Care Confirmation (serious, cosmetic) -> [Date Confirmation, Category Extraction] | Self"""
    logger.debug(f"{message.from_user.full_name} in {await state.get_state()}")

    if ag_level := await extract_agreement(message.text):
        if ag_level == Agreement.YES:
            date = datetime.now()
            await state.update_data(date=date)
            await message.answer(
                f"Подойдёт ли вам данное время записи: {date.isoformat(timespec='minutes')}?"
            )
            await state.set_state(states.date_confirmation)
        else:
            await message.answer(
                "Тогда давайте заново. Вы обратились из-за какой-то серьёзной проблемы?"
            )
            await state.set_state(states.care_category)
    else:
        await message.answer(GENERIC_RESPONSE_REPEAT)


@router.message(states.date_confirmation)
async def resolve_date_confirmation(message: Message, state: FSMContext) -> None:
    """Date Confirmation -> [Number Extraction, Self, End] | Self"""
    logger.debug(f"{message.from_user.full_name} in {await state.get_state()}")

    if ag_level := await extract_agreement(message.text):
        if ag_level == Agreement.YES:
            await message.answer(
                "Нам необходим ваш номер телефона. Мы напомним вас о записи за пару часов до назначенного времени."
            )
            await state.set_state(states.number_extraction)
        elif ag_level == Agreement.UNSURE:
            await message.answer(
                "К сожалению, на другое время не получится. Записать всё таки или нет?"
            )
        else:
            await message.answer(
                "К сожалению, мы не можем предоставить вам другое время записи, попробуйте в другой раз."
            )
            await state.set_state(states.end)
    else:
        await message.answer(GENERIC_RESPONSE_REPEAT)


@router.message(states.number_extraction)
async def resolve_number(message: Message, state: FSMContext) -> None:
    """Number Extraction -> Final Confirmation | Self"""
    logger.debug(f"{message.from_user.full_name} in {await state.get_state()}")

    if phone := await extract_number(message.text):
        data = await state.update_data(phone=phone)
        await message.answer(PROTOCOL.format(**data))
        await state.set_state(states.final_confirmation)
    else:
        await message.answer(GENERIC_RESPONSE_REPEAT)


@router.message(states.final_confirmation)
async def resolve_final_confirmation(message: Message, state: FSMContext) -> None:
    """Final Confirmation -> [End, Self, End] | Self"""
    logger.debug(f"{message.from_user.full_name} in {await state.get_state()}")

    if ag_level := await extract_agreement(message.text):
        if ag_level == Agreement.YES:
            await message.answer("Ждём вас и всего доброго.")
            await state.set_state(states.end)
        elif ag_level == Agreement.UNSURE:
            await message.answer("Решайтесь. Записать всё таки или нет?")
        else:
            await message.answer("Тогда до встречи.")
            await state.set_state(states.end)
    else:
        await message.answer(GENERIC_RESPONSE_REPEAT)


async def main() -> None:
    await init_intents()

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
