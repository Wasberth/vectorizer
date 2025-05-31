from flask import render_template, url_for, session, redirect, request
from decos import route
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv
load_dotenv()

@route('/vector/<filename>')
def index_page(filename):
    return render_template(f'index.html', stylesheets=['bootstrap.min'], scripts=['bootstrap.min'])
