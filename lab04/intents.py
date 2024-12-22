from enum import Enum
from .llm import AsyncPhi3

NAME_EXTRACT_SYS = (
    "You are tasked with extracting the name (first name) from the input text. "
    "If text contains more than one name, pick one that is most likely user's. "
    'If there is no name, respond only with "null". '
)

AGGREMENT_EXTRACT_SYS = (
    "You are tasked with understanding the level of aggreement in the input text. "
    'If user agrees, respond with "Y". If not, respond with "N". '
    'If it is unclear, respond with "U". You must respond with one of these letters.'
)

SERIOUS_CARE_SYS = (
    "You need to determine what is wrong with user's oral cavity from input text. "
    "Possible problems: 1 - Toothache, 2 - Problem with gums, 3 - Remove tooth, 0 - Unknown/something else/no problem. "
    "You must respond with corresponding digit."
)

COSMETIC_CARE_SYS = (
    "You need to determine what user what kind of dental service user needs from input text. "
    "Possible problems: 1 - Align or straighten teeth (for example with braces), 2 - Tooth whitening, 0 - Unknown/something else. "
    "You must respond with corresponding digit."
)

PHONE_EXTRACT_SYS = (
    "You are tasked with extracting the phone number from the input text. "
    "If text contains more than one, pick one that most likely belongs to the user. "
    "You must respond with a phone number itself, without any additional symbols. "
    'If there is no phone number, respond with "null".'
)

model: AsyncPhi3 = None


async def init_intents():
    """Loads a model that powers intents into global scope."""
    global model
    model = AsyncPhi3()


async def extract_name(text: str) -> str | None:
    """Extracts user's name from text or None."""

    global model

    extracted = await model.invoke(text, NAME_EXTRACT_SYS, max_new_tokens=16)

    if extracted in ["null", '"null"', ""]:
        return None
    else:
        return extracted


class Agreement(Enum):
    NO = 1
    YES = 2
    UNSURE = 3


async def extract_agreement(text: str) -> Agreement | None:
    """Extracts user's trinary response or None from text."""

    global model

    if extracted := await model.invoke(text, AGGREMENT_EXTRACT_SYS, max_new_tokens=2):
        extracted = extracted.lower()

        if extracted == "y":
            return Agreement.YES
        elif extracted == "n":
            return Agreement.NO
        elif extracted == "u":
            return Agreement.NO

    return None


class Problem(Enum):
    TEETHPAIN = 1
    GUMPAIN = 2
    LOOSETEETH = 3
    WHITEN = 4
    CORRECT = 5
    UNKNOWN = 0


async def extract_serious_care(text: str) -> Problem | None:
    """Extracts a type of user's health problem or None from text."""
    global model

    if extracted := await model.invoke(text, SERIOUS_CARE_SYS, max_new_tokens=2):
        if extracted == "1":
            return Problem.TEETHPAIN
        elif extracted == "2":
            return Problem.GUMPAIN
        elif extracted == "3":
            return Problem.LOOSETEETH
        elif extracted == "0":
            return Problem.UNKNOWN

    return None


async def extract_cosmetic_care(text: str) -> Problem | None:
    """Extracts a type of service user wants or None from text."""
    global model

    if extracted := await model.invoke(text, COSMETIC_CARE_SYS, max_new_tokens=2):
        if extracted == "1":
            return Problem.CORRECT
        elif extracted == "2":
            return Problem.WHITEN
        elif extracted == "0":
            return Problem.UNKNOWN

    return None


async def extract_number(text: str) -> str | None:
    """Extract user's phone number or None from text."""
    global model

    extracted = await model.invoke(text, PHONE_EXTRACT_SYS, max_new_tokens=16)

    if extracted in ["null", '"null"', ""]:
        return None
    else:
        return extracted
