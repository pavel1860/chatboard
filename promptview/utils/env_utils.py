





def is_running_in_jupyter():
    """
    Checks if the code is running inside a Jupyter Notebook or IPython kernel.
    """
    try:
        # The get_ipython function is available in the IPython/Jupyter environment
        # but not in a standard Python interpreter.
        shell = get_ipython().__class__.__name__
        if 'ZMQInteractiveShell' in shell:
            # Jupyter notebook or qtconsole
            return True
        elif 'TerminalInteractiveShell' in shell:
            # Terminal IPython
            return True
        else:
            # Other interactive environments
            return False
    except NameError:
        # Standard Python interpreter
        return False

if is_running_in_jupyter():
    print("Running in a Jupyter Notebook.")
else:
    print("Running in a standard Python shell.")
    
    
    
    
    
    
def try_apply_nest_asyncio():
    if is_running_in_jupyter():
        import nest_asyncio
        nest_asyncio.apply()