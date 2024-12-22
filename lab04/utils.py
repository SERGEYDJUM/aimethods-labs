from enum import StrEnum
from .intents import Problem


class Specialist(StrEnum):
    THERAPIST = "Терапевт"
    SURGEON = "Хирург"
    ORTHODONT = "Ортодонт"
    PARADONT = "Парадонтолог"
    HYGENIST = "Гигиенист"


class CareType(StrEnum):
    CURE = "Лечение"
    REMOVAL = "Удаление"
    CORRECTION = "Коррекция зубного ряда"
    GUMCURE = "Лечение дёсен"
    WHITENING = "Отбеливание зубов"
    EXAMINATION = "Обследование"


GENERIC_RESPONSE_REPEAT = "К сожалению, я не понимаю. Можете перефразировать?"
"""This is the main error phrase."""

PROTOCOL = """
Информация о записи:

Имя пациента: {user_name}
Телефон: {phone}
Врач: {specialist}
Направление: {care_type}

Записать?
"""


def problem_resolve(problem: Problem) -> tuple[Specialist, CareType, str]:
    """Maps the extracted problem to correspoding booking information.

    Returns:
        tuple[Specialist, CareType, str]: (doctor, care variant, the text for sending to user).
    """

    specialist = Specialist.THERAPIST
    care = CareType.EXAMINATION
    answer = "Я запишу вас к стоматологу-терапевту, который определит проблему и направит вас к нужному специалисту. Хорошо?"

    if problem == Problem.TEETHPAIN:
        specialist = Specialist.THERAPIST
        care = CareType.CURE
        answer = "Я направлю вас на лечение зубов к стоматологу-терапевту, хорошо?"
    elif problem == Problem.GUMPAIN:
        specialist = Specialist.PARADONT
        care = CareType.GUMCURE
        answer = "Я запишу вас к пародонтологу, который вылечит ваши десна, хорошо?"
    elif problem == Problem.LOOSETEETH:
        specialist = Specialist.SURGEON
        care = CareType.REMOVAL
        answer = (
            "Зарезервировать стоматолога-хирурга, чтобы он вырвал ваш проблемный зуб?"
        )
    elif problem == Problem.WHITEN:
        specialist = Specialist.HYGENIST
        care = CareType.WHITENING
        answer = "Направить вас к стоматологу-гигиенисту?"
    elif problem == Problem.CORRECT:
        specialist = Specialist.ORTHODONT
        care = CareType.CORRECTION
        answer = "Направить вас к ортодонту для исправления прикуса?"

    return (specialist, care, answer)
