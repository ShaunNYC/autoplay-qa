"""AutoPlay QA public surface with an explicit live-inference boundary."""

__version__ = "0.1.0"

from .game import GameAdapter, KeyGateGame
from .remote import RemoteGameAdapter

__all__ = ("GameAdapter", "KeyGateGame", "RemoteGameAdapter")
