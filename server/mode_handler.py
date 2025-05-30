import os
from dotenv import load_dotenv

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

def is_dev_mode():
    """
    Chequea si la variable "dev" en el .env es "true".

    :return: True si "dev" es "true", de lo contrario False.
    """
    return os.getenv("dev", "false").lower() == "true"

def get_url(service):
    """
    Obtiene la url del servicio dependiendo si estamos en dev o en prod

    :param str service: El nombre del servicio del que se requiere obtener su url
    :return: Url del servicio
    :rtype: str | None
    """

    if is_dev_mode():
        port = os.getenv(f'{service}_dev')
        if not port:
            print("El servicio no tiene un puerto asignado en dev")
            return None
        return f'http://localhost:{port}'
    
    return os.getenv(f"{service}_prod")

