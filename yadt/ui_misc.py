import os
import pathlib

import gradio as gr

from injector import inject, singleton

from yadt import ui_utils

from yadt.configuration import Configuration
from yadt.db_dataset import DatasetDB
from yadt.ui_shared import SharedState

@singleton
class MiscPage:
    @inject
    def __init__(self, configuration: Configuration, dataset_db: DatasetDB, shared_state: SharedState):
        self._configuration = configuration
        self._dataset_db = dataset_db
        self._shared_state = shared_state


    def _database_size(self):
        return os.stat(self._dataset_db.path).st_size

    def _dataset_cache_for_repo_name(self):
        return sorted(
            [ui_utils.NO_DROPDOWN_SELECTION] + self._dataset_db.get_dataset_cache_for_repo_name(),
            key=ui_utils.natural_sort,
        )

    def _dataset_cache_usage_for_repo_name(self):
        return sorted(
            [ [row['repo_name'], ui_utils.human_readable_bytes(row['bytes'])] for row in self._dataset_db.get_dataset_cache_usage_for_repo_name() ],
            key=lambda r: ui_utils.natural_sort(r[0]),
        )

    def _dataset_cache_for_dataset(self):
        return sorted(
            [ui_utils.NO_DROPDOWN_SELECTION] + [ row or 'UNKNOWN' for row in self._dataset_db.get_dataset_cache_for_dataset() ],
            key=ui_utils.natural_sort,
        )

    def _dataset_cache_usage_for_dataset(self):
        return sorted(
            [ [row['dataset'] or 'UNKNOWN', ui_utils.human_readable_bytes(row['bytes'])] for row in self._dataset_db.get_dataset_cache_usage_for_dataset() ],
            key=lambda r: ui_utils.natural_sort(r[0]),
        )

    def _reset_database(self):
        self._dataset_db.reset()

    def _drop_dataset_cache_for_repo_name(self, repo_name: str):
        if repo_name == ui_utils.NO_DROPDOWN_SELECTION:
            pass
        else:
            self._dataset_db.delete_dataset_cache_by_repo_name(repo_name)
    
    def _drop_dataset_cache_for_dataset(self, dataset: str):
        if dataset == ui_utils.NO_DROPDOWN_SELECTION:
            pass
        else:
            if dataset == 'UNKNOWN':
                dataset = None

            self._dataset_db.delete_dataset_cache_by_dataset(dataset)

    def _cache_folder_size(self):
        def dutree(p: pathlib.Path):
            if p.is_file():
                return p.stat().st_size
            elif p.is_dir():
                return sum([ dutree(p_child) for p_child in p.iterdir() ])
            else:
                raise AssertionError(f"Cannot calculate size since it's neither a file nor a folder: {str(p)}")

        return dutree(self._configuration.cache_folder)

    def _drop_cache_folder(self):
        def rmtree(p: pathlib.Path):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                for p_child in p.iterdir():
                    rmtree(p_child)
                p.rmdir()
            else:
                raise AssertionError(f"Cannot remove path since it's neither a file nor a folder: {str(p)}")

        p = pathlib.Path(self._configuration.cache_folder)
        for p_child in p.iterdir():
            rmtree(p_child)


    def ui(self):
        with gr.Blocks() as page:
            with gr.Row():
                with gr.Column():
                    dataset_db_title = gr.HTML(value = '<h3>Dataset DB</h3>')
                    
                    with gr.Column(variant="panel"):
                        with gr.Row(equal_height=True):
                            dataset_cache_for_repo_name_dropdown = gr.Dropdown(choices=self._dataset_cache_for_repo_name(), value=ui_utils.NO_DROPDOWN_SELECTION, interactive=True, label="Clear model cache", scale=1)
                            dataset_cache_for_repo_name_clear = gr.Button(value="Clear", variant="primary", scale=0)

                        with gr.Row(equal_height=True):
                            dataset_cache_for_dataset_dropdown = gr.Dropdown(choices=self._dataset_cache_for_dataset(), value=ui_utils.NO_DROPDOWN_SELECTION, interactive=True, label="Clear dataset cache", scale=1)
                            dataset_cache_for_dataset_clear = gr.Button(value="Clear", variant="primary", scale=0)

                    with gr.Column(variant="panel"):
                        with gr.Row():
                            vacuum = gr.Button(value="Vacuum")
                            refresh = gr.Button(value="Refresh", variant="primary")

                    with gr.Column(variant="panel"):
                        reset = gr.Button(value="Reset")
                    gr.HTML('<p style="margin-top: -1em"><i>WARNING: resetting the database will drop all caches and settings</i></p>')

                    cache_folder_title = gr.HTML(value = '<h3>Cache folder</h3>')
                    with gr.Column(variant="panel"):
                        with gr.Row():
                            clear_cache = gr.Button(value="Clear")
                            refresh_cache = gr.Button(value="Refresh", variant="primary")
                    gr.HTML('<p style="margin-top: -1em"><i>WARNING: clearing the cache folder will remove all cached images</i></p>')

                with gr.Column():
                    gr.HTML(value = f'<h3>Database statistics</h3>')

                    with gr.Column(variant="panel"):
                        gr.HTML('<p>Cache usage by model</p>')
                        cache_usage_by_model = gr.DataFrame(
                            headers=['Model', 'Usage'],
                            value=self._dataset_cache_usage_for_repo_name(),
                        )

                    with gr.Column(variant="panel"):
                        gr.HTML('<p>Cache usage by dataset</p><p style="font-size: 0.8em"><i>Caches might be shared between datasets, so total cache usage might be lower</i></p>')
                        cache_usage_by_dataset = gr.DataFrame(
                            headers=['Dataset', 'Usage'],
                            value=self._dataset_cache_usage_for_dataset(),
                        )

        @gr.on(
            (page.load, refresh.click, self._shared_state.db_cleared.change, self._shared_state.cache_cleared.change),
            inputs=[dataset_db_title],
            outputs=[dataset_db_title],
        )
        def _dataset_db_title(previous_title: str):
            with ui_utils.gradio_warning():
                return f'<h3>Dataset DB ({ui_utils.human_readable_bytes(self._database_size())})</h3>'
            return previous_title

        @gr.on(
            (page.load, refresh_cache.click, self._shared_state.db_cleared.change, self._shared_state.cache_cleared.change),
            inputs=[cache_folder_title],
            outputs=[cache_folder_title],
        )
        def _cache_folder_title(previous_title: str):
            with ui_utils.gradio_warning():
                return f'<h3>Cache folder ({ui_utils.human_readable_bytes(self._cache_folder_size())})</h3>'
            return previous_title


        @gr.on(
            dataset_cache_for_repo_name_clear.click,
            inputs=[dataset_cache_for_repo_name_dropdown],
            outputs=[
                dataset_cache_for_repo_name_dropdown,
                cache_usage_by_model,
            ],
        )
        def _drop_dataset_cache_for_repo_name(repo_name: str):
            with ui_utils.gradio_warning():
                self._drop_dataset_cache_for_repo_name(repo_name)

            return [
                gr.Dropdown(choices=self._dataset_cache_for_repo_name()),
                self._dataset_cache_usage_for_repo_name(),
            ]

        @gr.on(
            dataset_cache_for_dataset_clear.click,
            inputs=[dataset_cache_for_dataset_dropdown],
            outputs=[
                dataset_cache_for_dataset_dropdown,
                cache_usage_by_dataset,
            ],
        )
        def _drop_dataset_cache_for_dataset(dataset: str):
            with ui_utils.gradio_warning():
                self._drop_dataset_cache_for_dataset(dataset)

            return [
                gr.Dropdown(choices=self._dataset_cache_for_dataset()),
                self._dataset_cache_usage_for_dataset(),
            ]

        @gr.on(
            (refresh.click, self._shared_state.db_cleared.change, self._shared_state.cache_cleared.change),
            outputs=[
                dataset_cache_for_repo_name_dropdown,
                dataset_cache_for_dataset_dropdown,
                cache_usage_by_model,
                cache_usage_by_dataset,
            ],
        )
        def _refresh_database():
            return [
                gr.Dropdown(choices=self._dataset_cache_for_repo_name()),
                gr.Dropdown(choices=self._dataset_cache_for_dataset()),
                self._dataset_cache_usage_for_repo_name(),
                self._dataset_cache_usage_for_dataset(),
            ]
        
        @gr.on(
            reset.click,
            inputs=[self._shared_state.db_cleared],
            outputs=[self._shared_state.db_cleared],
        )
        def _reset_database(previous_state):
            with ui_utils.gradio_warning():
                self._reset_database()

                gr.Info('Dataset database folder has been reset.')

                return self._shared_state.notify_state_change()
            
            return previous_state

        @gr.on(
            vacuum.click,
            inputs=[dataset_db_title],
            outputs=[dataset_db_title],
        )
        def _vacuum_database(previous_title: str):
            self._dataset_db.vacuum()
            return _dataset_db_title(previous_title)


        @gr.on(
            clear_cache.click,
            inputs=[self._shared_state.cache_cleared],
            outputs=[self._shared_state.cache_cleared],
        )
        def _drop_cache_folder(previous_state):
            with ui_utils.gradio_warning():
                self._drop_cache_folder()
                self._reset_database()

                gr.Info('Cache folder has been cleared.')

                return self._shared_state.notify_state_change()

            return previous_state

