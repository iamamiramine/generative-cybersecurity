from enum import Enum

class BaseEnum(Enum):
    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

    @classmethod
    def get_values(cls):
        return list(cls._value2member_map_.keys())