"""Voice Playground: compare XTTS-v2, Chatterbox V3 and VoxCPM2 on free Turkish text."""

from voxforge.playground.service import VARIANTS, PlaygroundError, PlaygroundService, Variant, describe
from voxforge.playground.workers import EngineManager, WorkerError

__all__ = ["VARIANTS", "EngineManager", "PlaygroundError", "PlaygroundService", "Variant", "WorkerError", "describe"]
