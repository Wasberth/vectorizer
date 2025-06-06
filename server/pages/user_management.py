from flask import render_template, url_for, session, redirect, request
from decos import route
import os
from dotenv import load_dotenv
from database_connector import database as db
from hashlib import sha256
from vacaException import VacaException
load_dotenv()

@route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            form = request.form
            values = (form.get('user'), sha256(form.get('password').encode('utf-8')).hexdigest())
            cursor = db.cursor()
            cursor.execute('SELECT idUsuario FROM Usuario WHERE usuario=%s AND contrasena=%s', values)
            resultado = cursor.fetchone()
            if resultado == None:
                return render_template(f'login.html', stylesheets=['bootstrap.min', 'customchanges'], scripts=['bootstrap.min'], error='Usuario o contraseña incorrectas')
            session['id_usuario'] = resultado[0]
            return redirect(url_for('index_page'))
        except Exception:
            raise VacaException(500, "Error de conexión, inténtalo de nuevo más tarde.")
    if 'success' in request.args:
        return render_template('login.html', stylesheets=['bootstrap.min', 'customchanges'], scripts=['bootstrap.min'], success=request.args['success'])
    return render_template('login.html', stylesheets=['bootstrap.min', 'customchanges'], scripts=['bootstrap.min'])

@route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index_page'))

@route('/register', methods=['GET', 'POST'])
def registro():
    if request.method == 'POST':
        try:
            form = request.form
            cursor = db.cursor()
            values = (form.get('user'), sha256(form.get('password').encode('utf-8')).hexdigest())
            cursor.execute('INSERT INTO Usuario (usuario, contrasena) values (%s, %s)', values)
            db.commit()
            if cursor.rowcount == 0:
                return render_template('registro.html', stylesheets=['bootstrap.min', 'customchanges'], scripts=['bootstrap.min', 'validate_register'])
            return redirect(url_for('login', success='Usuario registrado, ahora inicie sesión.'), code=307)
        except Exception:
            raise VacaException(500, "Error de conexión, inténtalo de nuevo más tarde.")
    return render_template('registro.html', stylesheets=['bootstrap.min', 'customchanges'], scripts=['bootstrap.min', 'validate_register'])