from typing import Tuple, Dict

from PIL import Image

from yadt import tagger_camie
from yadt import tagger_smilingwolf
from yadt import tagger_florence2_promptgen

class Predictor:
    def __init__(self):
        self.last_loaded_repo = None
        self.model: 'Predictor' = None

    def load_model(self, model_repo: str, **kwargs):
        if self.last_loaded_repo == model_repo:
            return
        
        if model_repo.startswith(tagger_smilingwolf.MODEL_REPO_PREFIX):
            from yadt.tagger_smilingwolf import Predictor
            self.model = Predictor()
            self.model.load_model(model_repo, **kwargs)
        elif model_repo.startswith(tagger_camie.MODEL_REPO_PREFIX):
            from yadt.tagger_camie import Predictor
            self.model = Predictor()
            self.model.load_model(model_repo, **kwargs)
        elif model_repo.startswith(tagger_florence2_promptgen.MODEL_REPO_PREFIX):
            from yadt.tagger_florence2_promptgen import Predictor
            self.model = Predictor()
            self.model.load_model(model_repo, **kwargs)
        else:
            raise AssertionError("Model is not supported: " + model_repo)
        
        self.last_loaded_repo = model_repo


    def predict(self, image: Image) -> Tuple[str, Dict[str, float], Dict[str, float], Dict[str, float]]:
        assert self.model is not None, "No model loaded"
        return self.model.predict(image)

default_repo = tagger_smilingwolf.EVA02_LARGE_MODEL_DSV3_REPO

dropdown_list = [
    tagger_smilingwolf.SWINV2_MODEL_DSV3_REPO,
    tagger_smilingwolf.CONV_MODEL_DSV3_REPO,
    tagger_smilingwolf.VIT_MODEL_DSV3_REPO,
    tagger_smilingwolf.VIT_LARGE_MODEL_DSV3_REPO,
    tagger_smilingwolf.EVA02_LARGE_MODEL_DSV3_REPO,
    tagger_smilingwolf.MOAT_MODEL_DSV2_REPO,
    tagger_smilingwolf.SWIN_MODEL_DSV2_REPO,
    tagger_smilingwolf.CONV_MODEL_DSV2_REPO,
    tagger_smilingwolf.CONV2_MODEL_DSV2_REPO,
    tagger_smilingwolf.VIT_MODEL_DSV2_REPO,
    tagger_florence2_promptgen.FLORENCE2_PROMPTGEN_LARGE,
    tagger_florence2_promptgen.FLORENCE2_PROMPTGEN_BASE,
    tagger_camie.CAMIE_MODEL_FULL,
    tagger_camie.CAMIE_MODEL_INITIAL_ONLY,
]
