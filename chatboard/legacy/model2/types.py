

from enum import StrEnum
from chatboard.utils.model_utils import describe_enum


class ToolEnum(StrEnum):

    @classmethod
    def render(cls):
        return describe_enum(cls)
