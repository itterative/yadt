import traceback

import re
import contextlib

import gradio as gr

NO_DROPDOWN_SELECTION = '(None)'

@contextlib.contextmanager
def gradio_warning():
    try:
        yield
    except AssertionError as e:
        gr.Warning(str(e))
    except Exception as e:
        gr.Warning(str(e))
        traceback.print_exc()

def human_readable_bytes(size: int, units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']):
    for unit in units:
        if size < 1024:
            return f'{size:.2f} {unit}'
        
        size /= 1024
    else:
        return f'{size:.2f} {unit}'

_RE_NUMERIC_ = re.compile('([0-9]+)')
natural_sort = lambda key: [int(c) if c.isdigit() else c.lower() for c in _RE_NUMERIC_.split(key)]
