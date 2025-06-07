from flask import render_template, url_for, session, redirect, request, send_from_directory
from decos import route
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv
from PIL import Image
from pages._check_level_ import restricted
import hashlib
load_dotenv()

ALLOWED_EXTENSIONS = {'png', 'jpg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@route('/')
def index_page():
    """Renderiza la página principal"""
    return render_template(f'index.html', stylesheets=['dropzone', 'customchanges', 'bootstrap.min'], scripts=['bootstrap.bundle.min', 'drophandler_2', 'drophandler_1'])

@route('/cargar_imagen', methods=['POST'])
@restricted('user')
def get_img():
    if 'imagen' not in request.files:
        print('No hay imagen')
        return redirect(request.url)
    file = request.files['imagen']
    if file.filename == '':
        print('No hay imagen...')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        base_path = os.path.dirname(__file__) + os.environ['upload_path']

        hash_sha256 = hashlib.sha256()
        file.stream.seek(0)
        for chunk in iter(lambda: file.stream.read(4096), b""):
            hash_sha256.update(chunk)
        file.stream.seek(0)
        hash_name = hash_sha256.hexdigest() + filename[-4:]

        file.save(os.path.join(base_path, hash_name))
        image = Image.open(os.path.join(base_path, hash_name))
        image.save(os.path.join(base_path, 'original_'+hash_name))
        return redirect(url_for('preprocesamiento', name=hash_name))
    return redirect(request.url)