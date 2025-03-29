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

NO_DEFAULT = object()

def gradio_warning(*args, default=NO_DEFAULT):
    def _gradio_warning(fn):
        import traceback

        def fn_wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except AssertionError as e:
                gr.Warning(str(e))

                if default is not NO_DEFAULT:
                    return default
            except Exception as e:
                gr.Warning(str(e))
                traceback.print_exc()
                
                if default is not NO_DEFAULT:
                    return default

        return fn_wrapper

    if len(args) == 1 and callable(args[0]):
        return _gradio_warning(args[0])

    return _gradio_warning

def human_readable_bytes(size: int, units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']):
    for unit in units:
        if size < 1024:
            return f'{size:.2f} {unit}'
        
        size /= 1024
    else:
        return f'{size:.2f} {unit}'
