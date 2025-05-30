import re
from typing import List, Dict
from datetime import datetime

DATOS_PERSONALES_SECCION = '#Datos personales:{curp}|CURP;{fecha_nacimiento}|Fecha de Nacimiento>{genero}|Género;{nacionalidad}>>{lengua}|Habla lengua indígena>>{nivel_educativo} ({situacion_educativa})|Nivel Educativo'
DIRECCION_SECCION = '#Dirección:{estado_republica}|Estado>{delegacion_municipio}|Delegación/Municipio>>{colonia};{codigo_postal}>{direccion}|Dirección'
TALLAS_SECCION = '#Tallas:{playera}>{pantalon}>{calzado}'
BANCO_SECCION = '#Cuenta de Banco:{banco}>{cuenta_bancaria}|Cuenta'
APOYO_SECCION = '#Informarción apoyo:{beneficiario}>{monto_apoyo_convenio}|Monto de convenio'
CONTACTO_SECCION = '#Datos de Contacto:{email};{telefono_fijo};{telefono_movil}'
EVALUACION_EC1_SECCION = '#Evaluación del Evento "{evento}":{fecha}>{tipoEvaluacion}|Tipo de Evaluación>>{claridad};{comprension_lectora};{comprension_contenidos}|Comprensión de los contenidos>{eficiencia};{trabajo_equipo}|Trabajo en Equipo;{asistencia}>>{observaciones}'
EVALUACION_EC2_SECCION = '#Evaluación:{fecha}>{tipoEvaluacion}|Tipo de Evaluación>>{asistencia}>{relacion_comunidad}|Relación con la comunidad>{actitud}>>{observaciones}'
DATOS_PERSONALES_ALUMNO_SECCION = '#Datos personales:{curp}|CURP;{fecha_nacimiento}|Fecha de Nacimiento>{sexo}|Sexo'
INFO_EXTRA_ALUMNO_SECCION = '#Información extra:{cct}|CCT;{causa_baja}|Causa de baja'

CONVOCATORIA_TEMPLATE = '{nombre} {apellido} {folio}'+ DATOS_PERSONALES_SECCION + DIRECCION_SECCION \
    + TALLAS_SECCION + BANCO_SECCION + CONTACTO_SECCION

APOYO_TEMPLATE = '{nombre} {apellido} solicitada el {fecha_solicitud}' + DATOS_PERSONALES_SECCION + BANCO_SECCION + CONTACTO_SECCION \
    + APOYO_SECCION

EVALUACION_TEMPLATE = '{nombre} {apellido}' + DATOS_PERSONALES_SECCION + CONTACTO_SECCION

ALUMNO_TEMPLATE = '{nombre} {apellido1} {apellido2} {folio}' + DATOS_PERSONALES_ALUMNO_SECCION + INFO_EXTRA_ALUMNO_SECCION

def mapear_datos_convocatoria(data):
    """
    Mapea los datos de las convocatorias para que puedan ser usados en las cartas.
    :param list data: Lista de diccionarios con los datos de las convocatorias.
    :return: list
        Lista de diccionarios con los datos de las convocatorias mapeados.
    """

    for i in range(len(data)):
        convocatoria = data[i]
        
        apellido = convocatoria['apellido1']
        if 'apellido2' in convocatoria and convocatoria['apellido2']:
            apellido += ' ' + convocatoria['apellido2']
        data[i]['apellido'] = apellido

        
        cuenta_bancaria = convocatoria['cuenta_bancaria']
        if 'clabe' in convocatoria and convocatoria['clabe']:
            cuenta_bancaria = convocatoria['clabe']
        data[i]['cuenta_bancaria'] = cuenta_bancaria

        direccion = convocatoria['direccion']
        if 'num_exterior' in convocatoria and convocatoria['num_exterior']:
            direccion += ' No. ' + str(convocatoria['num_exterior'])
        else:
            direccion += ' S/N'

        if 'num_interior' in convocatoria and convocatoria['num_interior']:
            direccion += ' Interior ' + str(convocatoria['num_interior'])
        data[i]['direccion'] = direccion

        if 'fecha_nacimiento' in convocatoria and isinstance(convocatoria['fecha_nacimiento'], datetime):
            fecha = convocatoria['fecha_nacimiento'].strftime("%Y-%m-%d")
            data[i]['fecha_nacimiento'] = fecha


    return data

class Field:
    def __init__(self, label: str, value: str):
        self.label = label
        self.value = value

class Column:
    def __init__(self, fields: List[Field]):
        self.fields = fields

    def __iter__(self):
        return iter(self.fields)

class Row:
    def __init__(self, columns: List[Column]):
        self.columns = columns

    def __iter__(self):
        return iter(self.columns)

class Section:
    def __init__(self, title: str, rows: List[Row]):
        self.title = title
        self.content = rows

class Card:
    def __init__(self, title: str, sections: List[Section], raw_data: Dict[str, any] = None):
        self.title = title
        self.content = sections
        self.raw = raw_data

    def extend(self, extender: 'Card' | List['Card'] | Section | List[Section]):
        if isinstance(extender, Card):
            self.content.extend(extender.content)
        elif isinstance(extender, list):
            for item in extender:
                if isinstance(item, Card):
                    self.extend(item)
                elif isinstance(item, Section):
                    self.content.append(item)
                else:
                    raise ValueError('Invalid type for extender')
        elif isinstance(extender, Section):
            self.content.append(extender)
        else:
            raise ValueError('Invalid type for extender')

    @staticmethod
    def card_from_dict(template: str, *args, **kwargs):
        """
        Crea una carta a partir de un diccionario y un template
        :param str template: Template de la carta
            Los campos a reemplazar deben estar entre llaves
            El formato para la plantilla es el siguiente:
            título#seccion1:campo1;campo2|Alias>campo3;campo4>>campo5>campo6#seccion2:campo7;campo8>campo9
            # representa un cambio de sección, el texto antes de : es su título
            >> representa un salto de fila
            > representa un salto de columna
            ; separa campos
            | representa que el campo tiene un alias, el texto después de | es el alias
        :param dict args: Diccionario con los valores a reemplazar
        """

        def replace_placeholders(text: str, field_data: Dict[str, any]) -> str:
            """Reemplaza los placeholders en el texto usando el diccionario de datos."""
            def replacer(match):
                key = match.group(1)
                data = field_data.get(key)

                if data is None:
                    return 'No proporcionado'

                if type(data) == bool:
                    return 'Sí' if data else 'No'
                
                if data == 'false':
                    return 'No'
                
                if data == 'true':
                    return 'Sí'
                
                if data == '':
                    return 'No proporcionado'

                return str(data)
            
            return re.sub(r'\{(.*?)\}', replacer, text)

        # Merge args and kwargs into a single dictionary
        field_data = {}
        if len(args) == 1 and isinstance(args[0], dict):
            field_data.update(args[0])
        field_data.update(kwargs)

        # Parse the template string
        parts = template.split('#')
        title = replace_placeholders(parts[0], field_data)
        sections = []

        for section_part in parts[1:]:
            section_title, section_body = section_part.split(':', 1)
            section_title = replace_placeholders(section_title, field_data)

            rows = []
            for row in section_body.split('>>'):
                cols = []
                for col in row.split('>'):
                    fields = []
                    for field in col.split(';'):
                        if '|' in field:
                            value, label = field.split('|', 1)
                        else:
                            value, label = field, field
                            label = label.replace('_', ' ')
                            label = re.sub(r'\{(.*?)\}', r'\1', label).title()
                            

                        value = replace_placeholders(value, field_data)
                        label = replace_placeholders(label, field_data)
                        fields.append(Field(label, value))
                    cols.append(Column(fields))
                rows.append(Row(cols))
            sections.append(Section(section_title, rows))

        return Card(title, sections, field_data)
    
