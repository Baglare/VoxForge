"""Quality Director + Voice Takes (VF-018): UI- and engine-independent services.

- ``checks``: explainable rules (audio measurements vs ASR opinion, confidence levels);
- ``asr``: faster-whisper worker client (eval environment, CPU, no VRAM);
- ``director``: ``QualityDirector.check(wav, spoken_text, ...)`` -> report;
- ``takes``: take groups, alternatives with new seeds, ranking, selection and undo.

The Playground uses them through ``voxforge.playground.quality``; Studio (VF-019) can
plug its own synthesizer into ``TakeDirector`` with the same contract.
"""

from voxforge.quality.checks import CHECKER_VERSION, describe_report, evaluate, summarize
from voxforge.quality.director import QualityDirector
from voxforge.quality.takes import TakeDirector, TakeError, TakeStore

__all__ = [
    "CHECKER_VERSION",
    "QualityDirector",
    "TakeDirector",
    "TakeError",
    "TakeStore",
    "describe_report",
    "evaluate",
    "summarize",
]
