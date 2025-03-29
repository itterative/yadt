import argparse

import gradio as gr

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
    import tempfile

    from yadt import ui_image, ui_dataset, ui_misc

    args = parse_args()

    if args.device == "auto":
        import torch
        if torch.cuda.is_available():
            args.device = 'cuda:0'
        else:
            args.device = 'cpu'

    print('* Using device:', args.device)

    with tempfile.TemporaryDirectory(suffix='-yadt') as tempfolder:
        print('* Using temporary folder:', tempfolder)

        args.tempfolder = tempfolder

        with gr.Blocks(title=TITLE) as demo:
            with gr.Column():
                gr.Markdown(
                    value=f"<h1 style='text-align: center; margin-bottom: 1rem'>{TITLE}</h1>"
                )
                gr.Markdown(value=DESCRIPTION)

                with gr.Tabs():
                    with gr.Tab(label="Image"):
                        ui_image.ui(args)

                    with gr.Tab(label="Dataset"):
                        ui_dataset.ui(args)

                    with gr.Tab(label="Miscellaneous"):
                        ui_misc.ui(args)

        demo.queue(max_size=10)
        demo.launch(server_name=args.host, server_port=args.port, allowed_paths=[tempfolder])


if __name__ == "__main__":
    main()