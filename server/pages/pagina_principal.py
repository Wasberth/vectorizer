from flask import render_template, url_for, session, redirect, request
from decos import route
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv
from PIL import Image
load_dotenv()

ALLOWED_EXTENSIONS = {'png', 'jpg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@route('/')
def index_page():
    """Renderiza la página principal"""
    return render_template(f'index.html', stylesheets=['dropzone', 'customchanges', 'bootstrap.min'], scripts=['bootstrap.min', 'drophandler_1', 'drophandler_2'])

@route('/cargar_imagen', methods=['POST'])
def get_img():
    if 'imagen' not in request.files:
        print('No hay imagen')
        print(request.form)
        return redirect(request.url)
    file = request.files['imagen']
    if file.filename == '':
        print('No hay imagen...')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        base_path = os.path.dirname(__file__) + os.environ['upload_path']
        print(os.path.dirname(__file__))
        file.save(os.path.join(base_path, filename))
        image = Image.open(os.path.join(base_path, filename))
        image.save(os.path.join(base_path, 'original_'+filename))
        return redirect(url_for('preprocesamiento', name=filename))
    return redirect(request.url)