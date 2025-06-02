def route(path, methods=["GET"]):
    """Decorador para añadir la ruta y los métodos a los metadatos de la función."""
    def decorator(func):
        func.route = path
        func.methods = methods
        return func
    return decorator