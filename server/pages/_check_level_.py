from flask import redirect, url_for, session
from functools import wraps

def check_level(levels):
    """
    Verifica que el nivel sea el correcto, de lo contrario, lo redirige a una página compartida
    :param list|str levels: Lista de strings donde cada item marca el nivel de usuario permitido 
    """
    if type(levels) == str:
        levels = [levels]

    if 'user_id' not in session:
        return redirect(url_for('login'))
    if session['nivel'] not in levels:
        # TODO: Diseñar una página que tenga los vínculos únicos de cada usuario
        return redirect(url_for('wellcome'))
    
    return None

# Decorador para agregar a la función    
def restricted(levels):
    """
    Decorador que valida si el usuario tiene acceso a cierta página
    Si no tiene acceso, lo manda a una página global
    :param list|str levels: String o lista de strings donde cada item marca el nivel de usuario permitido 
    """
    def decorator(func):
        nonlocal levels

        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            Esta función es la que se ejecuta, aunque toma el mismo nombre
            """
            redirect_page = check_level(levels)
            if redirect_page is not None:
                return redirect_page
            
            return func(*args, **kwargs)
        
        if type(levels) == str:
            levels = [levels]
        wrapper.restricted = levels
        return wrapper
    
    return decorator
