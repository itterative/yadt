import gradio as gr

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
