def decompose_form(dict: dict[str, str]) -> list[dict[str, str]]:
    """
    Procesa un diccionario para desacoplar valores relacionados por identificadores numéricos y los agrupa en objetos individuales.

    :param dict:
        El diccionario de entrada con llaves que pueden terminar en números.
    :return:
        Lista de objetos desacoplados con llaves comunes y específicas por identificador.
    """
    # Diccionario para las claves comunes
    comunes = {}

    # Diccionario para agrupar las claves con números
    agrupados = {}

    for clave, valor in dict.items():
        if clave[-1].isdigit():  # Verifica si la clave termina en un número
            # Busca el índice del último guion bajo antes del número, si existe
            base_clave = clave.rstrip("0123456789").rstrip("_")
            numero = clave[len(base_clave):].lstrip("_")

            if numero not in agrupados:
                agrupados[numero] = {}
            agrupados[numero][base_clave] = valor
        else:
            comunes[clave] = valor

    # Crea la lista de objetos desacoplados
    desacoplados = []
    for _, grupo in agrupados.items():
        objeto = comunes.copy()  # Copia las claves comunes a cada objeto
        objeto.update(grupo)    # Agrega las claves específicas del grupo
        desacoplados.append(objeto)

    return desacoplados

if __name__ == "__main__":
    # Ejemplo de uso
    diccionario = {
        "fecha_evaluacion": "2024-05-05",
        "nombre1": "Juan Perez",
        "calificacion1_1": "10",
        "calificacion2_1": "10",
        "calificacion3_1": "10",
        "nombre2": "María Sánchez",
        "calificacion1_2": "7",
        "calificacion2_2": "8",
        "calificacion3_2": "9",
        "calificacion4_2": "10"
    }

    resultado = decompose_form(diccionario)
    print(resultado)
