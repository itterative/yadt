import gradio as gr

from injector import inject, singleton
from PIL import Image

from yadt.configuration import Configuration
from yadt.tagger_shared import Predictor

from yadt import tagger_shared
from yadt import process_prediction
from yadt import ui_utils


@singleton
class ImagePage:
    @inject
    def __init__(self, configuration: Configuration, predictor: Predictor):
        self._configuration = configuration
        self._predictor = predictor

    def _predict_image(
            self,
            image: Image,
            model_repo: str,
            general_thresh: float,
            general_mcut_enabled: bool,
            character_thresh: float,
            character_mcut_enabled: bool,
            replace_underscores: bool,
            trim_general_tag_dupes: bool,
            escape_brackets: bool,
    ):
        assert image is not None, "No image selected"

        self._predictor.load_model(model_repo, device=self._configuration.device)

        return process_prediction.post_process_prediction(
            *self._predictor.predict(image),
            general_thresh, general_mcut_enabled, character_thresh, character_mcut_enabled,
            replace_underscores, trim_general_tag_dupes, escape_brackets,
        )
    

    def ui(self):
        with gr.Row():
            with gr.Column(variant="panel"):
                image = gr.Image(type="pil", image_mode="RGBA", label="Input")
                model_repo = gr.Dropdown(
                    tagger_shared.dropdown_list,
                    value=tagger_shared.default_repo,
                    label="Model",
                )
                with gr.Row():
                    general_thresh = gr.Slider(
                        0,
                        1,
                        step=self._configuration.score_slider_step,
                        value=self._configuration.score_general_threshold,
                        label="General Tags Threshold",
                        scale=3,
                    )
                    general_mcut_enabled = gr.Checkbox(
                        value=False,
                        label="Use MCut threshold",
                        scale=1,
                    )
                with gr.Row():
                    character_thresh = gr.Slider(
                        0,
                        1,
                        step=self._configuration.score_slider_step,
                        value=self._configuration.score_character_threshold,
                        label="Character Tags Threshold",
                        scale=3,
                    )
                    character_mcut_enabled = gr.Checkbox(
                        value=False,
                        label="Use MCut threshold",
                        scale=1,
                    )
                with gr.Row():
                    replace_underscores = gr.Checkbox(
                        value=True,
                        label="Replace underscores with spaces",
                        scale=1,
                    )
                    trim_general_tag_dupes = gr.Checkbox(
                        value=True,
                        label="Trim duplicate general tags",
                        scale=1,
                    )
                    escape_brackets = gr.Checkbox(
                        value=True,
                        label="Escape brackets (for webui)",
                        scale=1,
                    )
                with gr.Row():
                    clear = gr.ClearButton(
                        components=[
                            image,
                            model_repo,
                            general_thresh,
                            general_mcut_enabled,
                            character_thresh,
                            character_mcut_enabled,
                        ],
                        variant="secondary",
                        size="lg",
                    )
        
                    submit = gr.Button(value="Submit", variant="primary", size="lg")
                    
            with gr.Column():
                sorted_general_strings = gr.Textbox(label="Output (string)", placeholder="Press the submit button to see the model outputs")

                with gr.Column(variant="panel"):
                    rating = gr.Label(label="Rating")
                    character_res = gr.Label(label="Output (characters)")
                    general_res = gr.Label(label="Output (tags)")
                    clear.add(
                        [
                            sorted_general_strings,
                            rating,
                            character_res,
                            general_res,
                        ]
                    )

        @gr.on(
            submit.click,
            inputs=[
                image,
                model_repo,
                general_thresh,
                general_mcut_enabled,
                character_thresh,
                character_mcut_enabled,
                replace_underscores,
                trim_general_tag_dupes,
                escape_brackets,
            ],
            outputs=[sorted_general_strings, rating, general_res, character_res],
        )
        def _predict_image(
                image: Image,
                model_repo: str,
                general_thresh: float,
                general_mcut_enabled: bool,
                character_thresh: float,
                character_mcut_enabled: bool,
                replace_underscores: bool,
                trim_general_tag_dupes: bool,
                escape_brackets: bool,
        ):
            with ui_utils.gradio_warning():
                return self._predict_image(
                    image,
                    model_repo,
                    general_thresh,
                    general_mcut_enabled,
                    character_thresh,
                    character_mcut_enabled,
                    replace_underscores,
                    trim_general_tag_dupes,
                    escape_brackets,
                )
            
            return '', {}, {}, {}
