"""Useful class, function and type definitions """
import enum

AgentID = int


class Outcomes(enum.Enum):
    """Outcomes from a POSG environment """
    LOSS = -1
    DRAW = 0
    WIN = 1
    NA = None

    def __str__(self):
        return self.name
