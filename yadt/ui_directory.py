import os
import gradio as gr

from PIL import Image

from yadt import tagger_shared
from yadt import process_prediction
from yadt import ui_utils

@ui_utils.gradio_error
def predict_folder(
        folder: str,
        model_repo: str,
        general_thresh: float,
        general_mcut_enabled: bool,
        character_thresh: float,
        character_mcut_enabled: bool,
        replace_underscores: bool,
        trim_general_tag_dupes: bool,
        escape_brackets: bool,
        overwrite_current_caption: bool,
        progress = gr.Progress(),
):
    tagger_shared.predictor.load_model(model_repo)

    files = os.listdir(folder)
    files = list(filter(lambda f: not f.endswith('.txt') and not f.endswith('.npz'), files))

    all_count = 0
    all_rating = dict()
    all_character_res = dict()
    all_general_res = dict()

    
    for index, file in progress.tqdm(list(enumerate(files))):
        try:
            image = Image.open(folder + '/' + file).convert("RGBA")
        except Exception as e:
            continue

        sorted_general_strings, rating, general_res, character_res = \
            process_prediction.post_process_prediction(
                *tagger_shared.predictor.predict(image),
                general_thresh, general_mcut_enabled, character_thresh, character_mcut_enabled,
                replace_underscores, trim_general_tag_dupes, escape_brackets,
            )

        all_count += 1

        for k in rating.keys():
            all_rating[k] = all_rating.get(k, 0) + rating[k]

        for k in character_res.keys():
            all_character_res[k] = all_character_res.get(k, 0) + 1
        
        for k in general_res.keys():
            all_general_res[k] = all_general_res.get(k, 0) + 1

        caption_file = file[:file.rindex('.')] + '.txt'
        caption_file_path = folder + '/' + caption_file

        if not os.path.exists(caption_file_path) or overwrite_current_caption:
            with open(folder + '/' + caption_file, 'w') as f:
                f.write(sorted_general_strings)

    for k in all_rating.keys():
        all_rating[k] = all_rating[k] / all_count

    for k in all_character_res.keys():
        all_character_res[k] = all_character_res[k] / all_count
    
    for k in all_general_res.keys():
        all_general_res[k] = all_general_res[k] / all_count

    return all_rating, all_general_res, all_character_res


def ui(args):
    with gr.Row():
        with gr.Column(variant="panel"):
            folder = gr.Textbox(label="Select folder:")
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
                overwrite_current_caption = gr.Checkbox(
                    value=False,
                    label="Overwrite existing captions",
                    scale=1,
                )
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
                        folder,
                        model_repo,
                        general_thresh,
                        general_mcut_enabled,
                        character_thresh,
                        character_mcut_enabled,
                        replace_underscores,
                        trim_general_tag_dupes,
                        escape_brackets,
                        overwrite_current_caption,
                    ],
                    variant="secondary",
                    size="lg",
                )
    
                submit = gr.Button(value="Submit", variant="primary", size="lg")
        
        with gr.Column(variant="panel"):
            rating = gr.Label(label="Rating")
            character_res = gr.Label(label="Output (characters)")
            general_res = gr.Label(label="Output (tags)")
            clear.add(
                [
                    rating,
                    character_res,
                    general_res,
                ]
            )

    submit.click(
        predict_folder,
        inputs=[
            folder,
            model_repo,
            general_thresh,
            general_mcut_enabled,
            character_thresh,
            character_mcut_enabled,
            replace_underscores,
            trim_general_tag_dupes,
            escape_brackets,
            overwrite_current_caption,
        ],
        outputs=[rating, general_res, character_res],
    )
