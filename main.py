import argparse
import pathlib

import gradio as gr

from injector import Injector

from yadt.configuration_injector import InjectorConfiguration
from yadt.ui_image import ImagePage
from yadt.ui_dataset import DatasetPage
# from yadt.ui_directory import DirectoryPage
from yadt.ui_misc import MiscPage
from yadt.ui_shared import SharedState

TITLE = "Yet Another Dataset Tagger"
DESCRIPTION = """
<center>models are 90% data, 10% training</center>
"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--score-slider-step", type=float, default=0.05)
    parser.add_argument("--score-general-threshold", type=float, default=0.35)
    parser.add_argument("--score-character-threshold", type=float, default=0.9)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        import torch
        if torch.cuda.is_available():
            args.device = 'cuda:0'
        else:
            args.device = 'cpu'

    print('* Using device:', args.device)

    cache_folder = pathlib.Path(__file__).parent / '.cache_save'
    cache_folder.mkdir(exist_ok=True)

    print('* Using cache folder:', cache_folder)

    injector = Injector(InjectorConfiguration(
        device=args.device,
        cache_folder=cache_folder,
        score_character_threshold=args.score_character_threshold,
        score_general_threshold=args.score_general_threshold,
        score_slider_step=args.score_slider_step
    ))

    with gr.Blocks(title=TITLE) as demo:
        _ = injector.get(SharedState)

        with gr.Column():
            gr.Markdown(value=f"<h1 style='text-align: center; margin-bottom: 1rem'>{TITLE}</h1>")
            gr.Markdown(value=DESCRIPTION)

            with gr.Tabs():
                with gr.Tab(label="Image"):
                    injector.get(ImagePage).ui()

                # with gr.Tab(label="Directory"):
                #     injector.get(DirectoryPage).ui()

                with gr.Tab(label="Dataset"):
                    injector.get(DatasetPage).ui()

                with gr.Tab(label="Miscellaneous"):
                    injector.get(MiscPage).ui()

    demo.queue(max_size=10)
    demo.launch(server_name=args.host, server_port=args.port, allowed_paths=[cache_folder])


if __name__ == "__main__":
    main()