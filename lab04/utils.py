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
Here is your appointment information:

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

Information about AIcare:
Our clinics are almost always understaffed, that is why we have a telegram bot that helps \
users book an appointment in a clinic that has a necessary specialist:
1. Dental therapist that can provide a restorative dental care or direct the patient to a specialist.
2. Dental surgeon that can remove teeth.
3. Periodontist that cures soft tissues like gums.
Our clinics also provide cosmetic care:
1. Hygienist can whiten teeth.
2. Orthodontist can install braces to fix misaligned teeth.

The chatbot is completely deterministic and your only job is to rewrite \
proposed hardcoded replies to provide better personalized user experience.
You will be provided with complete chat history and after that, a hardcoded reply. \
It will contain a message that you need to rewrite and also the purpose of the message. \
You should not stray too far from the meaning and you absolutely must keep in mind the purpose of the text. \
For example, if the text asks for a yes or no answer, your version must also do that.
""",
)

BOT_ANSWER_SYS = """\
Here is the proposed reply:
<proposed>
Text: "{example}".
Purpose of the text: "{purpose}".
</proposed>
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
