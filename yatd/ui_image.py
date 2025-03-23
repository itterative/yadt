import gradio as gr

from PIL import Image

from yatd import tagger_shared
from yatd import process_prediction
from yatd import ui_utils

@ui_utils.gradio_error
def predict(
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

    tagger_shared.predictor.load_model(model_repo)

    return process_prediction.post_process_prediction(
        *tagger_shared.predictor.predict(image),
        general_thresh, general_mcut_enabled, character_thresh, character_mcut_enabled,
        replace_underscores, trim_general_tag_dupes, escape_brackets,
    )


def ui(args):
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
                    step=args.score_slider_step,
                    value=args.score_general_threshold,
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
                    step=args.score_slider_step,
                    value=args.score_character_threshold,
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
                
        with gr.Column(variant="panel"):
            sorted_general_strings = gr.Textbox(label="Output (string)")
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

    submit.click(
        predict,
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
