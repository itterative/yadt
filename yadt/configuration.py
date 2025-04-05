import pathlib

from dataclasses import dataclass

@dataclass
class Configuration:
    device: str
    cache_folder: pathlib.Path
    score_slider_step: float
    score_general_threshold: float
    score_character_threshold: float
