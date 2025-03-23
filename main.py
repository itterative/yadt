import argparse

import gradio as gr

TITLE = "Booru Tagger"
DESCRIPTION = """
<center>models are 90% data, 10% training</center>
"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--score-slider-step", type=float, default=0.05)
    parser.add_argument("--score-general-threshold", type=float, default=0.35)
    parser.add_argument("--score-character-threshold", type=float, default=0.9)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main():
    from yatd import ui_image, ui_directory, ui_dataset

    args = parse_args()

    with gr.Blocks(title=TITLE) as demo:
        with gr.Column():
            gr.Markdown(
                value=f"<h1 style='text-align: center; margin-bottom: 1rem'>{TITLE}</h1>"
            )
            gr.Markdown(value=DESCRIPTION)

            with gr.Tabs():
                with gr.Tab(label="Image"):
                    ui_image.ui(args)

                with gr.Tab(label="Directory"):
                    ui_directory.ui(args)

                with gr.Tab(label="Dataset"):
                    ui_dataset.ui(args)

    demo.queue(max_size=10)
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
