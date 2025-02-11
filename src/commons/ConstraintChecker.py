from enum import Enum

class ConstraintChecker(Enum):
    SATISFIED = 1
    POSSIBLY_SATISFIED = 0.66
    POSSIBLY_VIOLATED = 0.33