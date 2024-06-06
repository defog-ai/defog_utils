from dataclasses import dataclass
import re

from .constants import idk_re_pattern
from .utils_feature import Features


@dataclass
class InstructionFeatures(Features):
    """
    Dataclass for tracking features extracted from the instructions.
    """

    _prefix: str = "instruction"

    add_alias_prefix: bool = False
    idk: bool = False


alias_re_pattern = re.compile(r"\balias(?:es)?\b", re.IGNORECASE)


def get_instruction_features(instructions: str) -> InstructionFeatures:
    """
    Extracts features from the instructions.
    `instructions` should be a single string containing the instructions.
    """
    features = InstructionFeatures()

    if re.search(alias_re_pattern, instructions):
        features.add_alias_prefix = True

    if re.search(idk_re_pattern, instructions):
        features.idk = True

    return features
