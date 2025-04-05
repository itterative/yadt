import random
import gradio as gr

from injector import singleton

@singleton
class SharedState:
    def __init__(self):
        self.db_cleared = gr.JSON(value=[], visible=False)
        self.cache_cleared = gr.JSON(value=[], visible=False)

    def notify_state_change(self):
        return gr.update(value=[random.random()])
