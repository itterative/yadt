import gradio as gr

NO_DROPDOWN_SELECTION = '(None)'

def gradio_error(fn):
    import traceback

    def fn_wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except AssertionError as e:
            raise gr.Error(str(e), print_exception=False) from e
        except Exception as e:
            traceback.print_exc()
            raise gr.Error(str(e), print_exception=False) from e
    
    return fn_wrapper

def human_readable_bytes(size: int, units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']):
    for unit in units:
        if size < 1024:
            return f'{size:.2f} {unit}'
        
        size /= 1024
    else:
        return f'{size:.2f} {unit}'
