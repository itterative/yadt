from typing import Tuple, Dict

from PIL import Image

from yatd import tagger_camie
from yatd import tagger_smilingwolf

class Predictor:
    def __init__(self):
        self.last_loaded_repo = None
        self.model: 'Predictor' = None

    def load_model(self, model_repo: str):
        if self.last_loaded_repo == model_repo:
            return
        
        if model_repo.startswith(tagger_smilingwolf.MODEL_REPO_PREFIX):
            from tagger_smilingwolf import Predictor
            self.model = Predictor()
            self.model.load_model(model_repo)
        elif model_repo.startswith(tagger_camie.MODEL_REPO_PREFIX):
            from tagger_camie import Predictor
            self.model = Predictor()
            self.model.load_model(model_repo)
        else:
            raise AssertionError("Model is not supported: " + model_repo)
        
        self.last_loaded_repo = model_repo


    def predict(self, image: Image) -> Tuple[str, Dict[str, float], Dict[str, float], Dict[str, float]]:
        assert self.model is not None, "No model loaded"
        return self.model.predict(image)

predictor = Predictor()

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
    tagger_camie.CAMIE_MODEL_FULL,
    tagger_camie.CAMIE_MODEL_INITIAL_ONLY,
]
