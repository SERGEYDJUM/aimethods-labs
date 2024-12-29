from enum import Enum
from loguru import logger
from .llm import AsyncGenericModel

NAME_EXTRACT_SYS = (
    "You are tasked with extracting the name (first name) from the input text. "
    "If text contains more than one name, pick one that is most likely user's. "
    'If there is no name, respond only with "null". '
    "You must respond with a name without any quotation marks."
)

AGGREMENT_EXTRACT_SYS = (
    "You are tasked with understanding the level of aggreement in the input text. "
    'If user agrees, respond with "Y". If not, respond with "N". '
    'If it is completely unclear, respond with "U". You must respond with one of these letters without quotation marks.'
)

SERIOUS_CARE_SYS = (
    "You need to determine what is wrong with user's oral cavity from input text. "
    "Possible problems: 1 - Toothache, 2 - Pain or bleeding in gums, 3 - Remove tooth, 0 - Unknown/something else/no problem. "
    "You must respond with corresponding digit without quotation marks."
)

COSMETIC_CARE_SYS = (
    "You need to determine what user what kind of dental service user needs from input text. "
    "Possible problems: 1 - Align or straighten teeth (for example with braces), 2 - Tooth whitening, 0 - Unknown/something else. "
    "You must respond with corresponding digit without quotation marks."
)

PHONE_EXTRACT_SYS = (
    "You are tasked with extracting the phone number from the input text. "
    "If text contains more than one, pick one that most likely belongs to the user. "
    "You must respond with a phone number itself, without any additional symbols. "
    'If there is no phone number, respond with "null" without quotation marks.'
)


async def extract_name(text: str, model: AsyncGenericModel) -> str | None:
    """Extracts user's name from text or None."""
    extracted = await model.invoke(text, NAME_EXTRACT_SYS, max_new_tokens=16)

    if extracted in ["null", '"null"', ""]:
        logger.warning(
            f"intent failure: type: Name, input: `{text}`, output: `{extracted}`"
        )
        return None
    else:
        return extracted


class Agreement(Enum):
    NO = 1
    YES = 2
    UNSURE = 3


async def extract_agreement(text: str, model: AsyncGenericModel) -> Agreement | None:
    """Extracts user's trinary response or None from text."""
    if extracted := await model.invoke(text, AGGREMENT_EXTRACT_SYS, max_new_tokens=2):
        extracted = extracted.lower()

        if extracted == "y":
            return Agreement.YES
        elif extracted == "n":
            return Agreement.NO
        elif extracted == "u":
            return Agreement.UNSURE

        logger.error(
            f"intent failure: type: Agreement, input: `{text}`, output: `{extracted}`"
        )
    return None


class Problem(Enum):
    TEETHPAIN = 1
    GUMPAIN = 2
    LOOSETEETH = 3
    WHITEN = 4
    CORRECT = 5
    UNKNOWN = 0


async def extract_serious_care(text: str, model: AsyncGenericModel) -> Problem | None:
    """Extracts a type of user's health problem or None from text."""
    if extracted := await model.invoke(text, SERIOUS_CARE_SYS, max_new_tokens=2):
        if extracted == "1":
            return Problem.TEETHPAIN
        elif extracted == "2":
            return Problem.GUMPAIN
        elif extracted == "3":
            return Problem.LOOSETEETH
        elif extracted == "0":
            return Problem.UNKNOWN

        logger.error(
            f"intent failure: type: SeriousCareType, input: `{text}`, output: `{extracted}`"
        )
    return None


async def extract_cosmetic_care(text: str, model: AsyncGenericModel) -> Problem | None:
    """Extracts a type of service user wants or None from text."""
    if extracted := await model.invoke(text, COSMETIC_CARE_SYS, max_new_tokens=2):
        if extracted == "1":
            return Problem.CORRECT
        elif extracted == "2":
            return Problem.WHITEN
        elif extracted == "0":
            return Problem.UNKNOWN

        logger.error(
            f"intent failure: type: CosmeticCareType, input: `{text}`, output: `{extracted}`"
        )
    return None


async def extract_number(text: str, model: AsyncGenericModel) -> str | None:
    """Extract user's phone number or None from text."""
    global models

    extracted = await model.invoke(text, PHONE_EXTRACT_SYS, max_new_tokens=20)

    if extracted in ["null", '"null"', ""]:
        logger.warning(
            f"intent failure: type: PhoneNumber, input: `{text}`, output: `{extracted}`"
        )
        return None
    else:
        return extracted
