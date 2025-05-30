def route(path, methods=["GET"]):
    """Decorador para añadir la ruta y los métodos a los metadatos de la función."""
    def decorator(func):
        func.route = path
        func.methods = methods
        return func
    return decorator

def nav(text):
    """Decorador para añadir el texto de navegación a los metadatos de la función."""
    def decorator(func):
        func.nav = text
        return func
    return decorator

def offline():
    """Decorador para marcar una página como offline."""
    def decorator(func):
        func.offline = True
        return func
    return decorator