import gradio as gr

from yadt import ui_utils
from yadt.dataset_db import db


def database_size():
    import os

    try:
        dataset_db_size = os.stat(db.path).st_size
    except:
        dataset_db_size = 0

    return f'<h3>Dataset DB ({ui_utils.human_readable_bytes(dataset_db_size)})</h3>'

def dataset_cache_for_repo_name():
    return sorted(
        [ui_utils.NO_DROPDOWN_SELECTION] + db.get_dataset_cache_for_repo_name(),
        key=ui_utils.natural_sort,
    )

def dataset_cache_usage_for_repo_name():
    return sorted(
        [ [row['repo_name'], ui_utils.human_readable_bytes(row['bytes'])] for row in db.get_dataset_cache_usage_for_repo_name() ],
        key=lambda r: ui_utils.natural_sort(r[0]),
    )

def dataset_cache_for_dataset():
    return sorted(
        [ui_utils.NO_DROPDOWN_SELECTION] + [ row or 'UNKNOWN' for row in db.get_dataset_cache_for_dataset() ],
        key=ui_utils.natural_sort,
    )

def dataset_cache_usage_for_dataset():
    return sorted(
        [ [row['dataset'] or 'UNKNOWN', ui_utils.human_readable_bytes(row['bytes'])] for row in db.get_dataset_cache_usage_for_dataset() ],
        key=lambda r: ui_utils.natural_sort(r[0]),
    )

@ui_utils.gradio_error
def vacuum_database():
    db.vacuum()
    
    return [
        database_size(),
    ]

@ui_utils.gradio_error
def refresh_database():
    return [
        database_size(),
        gr.Dropdown(choices=dataset_cache_for_repo_name()),
        gr.Dropdown(choices=dataset_cache_for_dataset()),
        dataset_cache_usage_for_repo_name(),
        dataset_cache_usage_for_dataset(),
    ]

@ui_utils.gradio_error
def reset_database():
    db.reset()
    return refresh_database()

@ui_utils.gradio_error
def drop_dataset_cache_for_repo_name(repo_name: str):
    if repo_name == ui_utils.NO_DROPDOWN_SELECTION:
        pass
    else:
        db.delete_dataset_cache_by_repo_name(repo_name)

    return [
        database_size(),
        gr.Dropdown(choices=dataset_cache_for_repo_name()),
        dataset_cache_usage_for_repo_name(),
    ]

@ui_utils.gradio_error
def drop_dataset_cache_for_dataset(dataset: str):
    if dataset == ui_utils.NO_DROPDOWN_SELECTION:
        pass
    else:
        if dataset == 'UNKNOWN':
            dataset = None

        db.delete_dataset_cache_by_dataset(dataset)

    return [
        database_size(),
        gr.Dropdown(choices=dataset_cache_for_dataset()),
        dataset_cache_usage_for_dataset(),
    ]


def ui(args):
    with gr.Row():
        with gr.Column():
            dataset_db_title = gr.HTML(value = database_size)
            
            with gr.Column(variant="panel"):
                with gr.Row(equal_height=True):
                    dataset_cache_for_repo_name_dropdown = gr.Dropdown(choices=dataset_cache_for_repo_name(), value=ui_utils.NO_DROPDOWN_SELECTION, interactive=True, label="Clear model cache", scale=1)
                    dataset_cache_for_repo_name_clear = gr.Button(value="Clear", variant="primary", scale=0)

                with gr.Row(equal_height=True):
                    dataset_cache_for_dataset_dropdown = gr.Dropdown(choices=dataset_cache_for_dataset(), value=ui_utils.NO_DROPDOWN_SELECTION, interactive=True, label="Clear dataset cache", scale=1)
                    dataset_cache_for_dataset_clear = gr.Button(value="Clear", variant="primary", scale=0)

            with gr.Column(variant="panel"):
                with gr.Row():
                    vacuum = gr.Button(value="Vacuum")
                    refresh = gr.Button(value="Refresh", variant="primary")

            with gr.Column(variant="panel"):
                reset = gr.Button(value="Reset")
            gr.HTML('<p style="margin-top: -1em"><i>WARNING: resetting the database will drop all caches and settings</i></p>')

        with gr.Column():
            gr.HTML(value = f'<h3>Database statistics</h3>')

            with gr.Column(variant="panel"):
                gr.HTML('<p>Cache usage by model</p>')
                cache_usage_by_model = gr.DataFrame(
                    headers=['Model', 'Usage'],
                    value=dataset_cache_usage_for_repo_name(),
                )

            with gr.Column(variant="panel"):
                gr.HTML('<p>Cache usage by dataset</p><p style="font-size: 0.8em"><i>Caches might be shared between datasets, so total cache usage might be lower</i></p>')
                cache_usage_by_dataset = gr.DataFrame(
                    headers=['Dataset', 'Usage'],
                    value=dataset_cache_usage_for_dataset(),
                )

    dataset_cache_for_repo_name_clear.click(
        drop_dataset_cache_for_repo_name,
        inputs=[dataset_cache_for_repo_name_dropdown],
        outputs=[
            dataset_db_title,
            dataset_cache_for_repo_name_dropdown,
            cache_usage_by_model,
        ]
    )

    dataset_cache_for_dataset_clear.click(
        drop_dataset_cache_for_dataset,
        inputs=[dataset_cache_for_dataset_dropdown],
        outputs=[
            dataset_db_title,
            dataset_cache_for_dataset_dropdown,
            cache_usage_by_dataset,
        ]
    )

    refresh.click(
        refresh_database,
        outputs=[
            dataset_db_title,
            dataset_cache_for_repo_name_dropdown,
            dataset_cache_for_dataset_dropdown,
            cache_usage_by_model,
            cache_usage_by_dataset,
        ],
    )

    reset.click(
        reset_database,
        outputs=[
            dataset_db_title,
            dataset_cache_for_repo_name_dropdown,
            dataset_cache_for_dataset_dropdown,
            cache_usage_by_model,
            cache_usage_by_dataset,
        ],
    )

    vacuum.click(
        vacuum_database,
        outputs=[dataset_db_title],
    )
