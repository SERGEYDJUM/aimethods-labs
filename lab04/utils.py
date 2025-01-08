from enum import StrEnum
from typing import Any
from .intents import Problem
from aiogram.types import Message
from aiogram.fsm.context import FSMContext
from .llm import LLMMessage, LLMRole


class Specialist(StrEnum):
    THERAPIST = "Therapist"
    SURGEON = "Dental Surgeon"
    ORTHODONT = "Orthodontist"
    PARADONT = "Periodontist"
    HYGENIST = "Hygienist"


class CareType(StrEnum):
    CURE = "Restoration"
    REMOVAL = "Removal"
    CORRECTION = "Correction"
    GUMCURE = "Soft Tissue Care"
    WHITENING = "Teeth Whitening"
    EXAMINATION = "Examination"


GENERIC_RESPONSE_REPEAT = "I don't understand. Can you repeat that?"
"""This is the main error phrase."""

PROTOCOL = """
Appointment information:

Patient's name: {user_name}
Phone Number: {phone}
Specialist: {specialist}
Care type: {care_type}

Is everything okay with it? I need your final confirmation.
"""

BOT_GLOBAL_SYS = LLMMessage(
    LLMRole.SYSTEM,
    """\
You work for a dental franchise in USA called AIcare.

Our clinics are almost always understaffed, that is why we have a telegram bot that helps \
users book an appointment in a clinic that has a necessary specialist:
1. Dental therapist that can provide a restorative dental care or direct the patient to a specialist.
2. Dental surgeon that can remove teeth.
3. Periodontist that cures soft tissues like gums.
Our clinics also provide cosmetic care:
1. Hygienist can whiten teeth.
2. Orthodontist can install braces to fix misaligned teeth.


Information for you:
1. The chatbot logic is completely deterministic: it will propose a reply for user's input message.
2. Your only job is to rewrite that proposed reply to provide better personalized user experience. \
3. You will be provided with complete chat history and after that, a hardcoded reply. \
It will contain a message that you must rewrite. More than that, it also contains the purpose of that proposed message to guide you. \
4. When rewriting, you absolutely must keep in mind the purpose of the proposed reply. \
For example, if the proposed reply to user input asks for a yes or no answer, your output can be anything, but it must also ask the yes or no question. \
This is important because your output must not interfere with deterministic chat flow.
""",
)

BOT_ANSWER_SYS = """\
Here is the hardcoded proposed reply: \"\"\"
{example}
\"\"\"
Purpose of the reply: \"\"\"
{purpose}
\"\"\"
"""


def problem_resolve(problem: Problem) -> tuple[Specialist, CareType, str]:
    """Maps the extracted problem to correspoding booking information.

    Returns:
        tuple[Specialist, CareType, str]: (doctor, care variant, the text for sending to user).
    """

    specialist = Specialist.THERAPIST
    care = CareType.EXAMINATION
    answer = "I will make an appointment for you to see a dental therapist who will determine your problem, ok?"

    if problem == Problem.TEETHPAIN:
        specialist = Specialist.THERAPIST
        care = CareType.CURE
        answer = "I will make an appointment for you to see a dental therapist who will cure your teeth, ok?"
    elif problem == Problem.GUMPAIN:
        specialist = Specialist.PARADONT
        care = CareType.GUMCURE
        answer = "I will make an appointment for you to see a periodontist who will cure your gums, ok?"
    elif problem == Problem.LOOSETEETH:
        specialist = Specialist.SURGEON
        care = CareType.REMOVAL
        answer = "I will make an appointment for you to see a dental surgeon who will remove your tooth, ok?"
    elif problem == Problem.WHITEN:
        specialist = Specialist.HYGENIST
        care = CareType.WHITENING
        answer = "I will make an appointment for you to see a dental hygienist who will make your teeth white, ok?"
    elif problem == Problem.CORRECT:
        specialist = Specialist.ORTHODONT
        care = CareType.CORRECTION
        answer = "I will make an appointment for you to see an orthodontist who will align your teeth, ok?"

    return (specialist, care, answer)
