import huggingface_hub

from PIL import Image

MODEL_REPO_PREFIX = "Camais03/" 

CAMIE_MODEL_FULL = "Camais03/camie-tagger"
CAMIE_MODEL_INITIAL_ONLY = "Camais03/camie-tagger (low vram/initial only)"

class Predictor:
    def __init__(self):
        self.model = None

    def download_model(self, full_model: bool):
        metadata_path = huggingface_hub.hf_hub_download(
            CAMIE_MODEL_FULL,
            'model/metadata.json',
        )

        if full_model:
            model_info_path = huggingface_hub.hf_hub_download(
                CAMIE_MODEL_FULL,
                'model/model_info_refined.json',
            )

            state_dict_path = huggingface_hub.hf_hub_download(
                CAMIE_MODEL_FULL,
                'model/model_refined.pt',
            )
        else:
            model_info_path = huggingface_hub.hf_hub_download(
                CAMIE_MODEL_FULL,
                'model/model_info_initial.json',
            )

            state_dict_path = huggingface_hub.hf_hub_download(
                CAMIE_MODEL_FULL,
                'model/model_initial_only.pt',
            )

        return metadata_path, model_info_path, state_dict_path

    def load_model(self, model_repo: str, **kwargs):
        full_model = model_repo == CAMIE_MODEL_FULL
        metadata_path, model_info_path, state_dict_path = self.download_model(full_model)

        device = kwargs.pop('device', 'cpu')

        from yadt.tagger_camie_model import load_model

        self.model, _, _ = load_model(
            '.',
            full=full_model,
            metadata_path=metadata_path,
            model_info_path=model_info_path,
            state_dict_path=state_dict_path,
            device=device,
        )


    def predict(self, image: Image):
        assert self.model is not None, "No model loaded"

        results = self.model.predict(image)
        tags = self.model.get_tags_from_predictions(results['predictions'], probabilities=results['refined_probabilities'])

        return dict(tags['rating']), dict(tags['general']), dict(tags['character'])


