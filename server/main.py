import os
import importlib.util
from flask import Flask, render_template, Response, session
import inspect
from werkzeug.exceptions import HTTPException
from dotenv import load_dotenv
import database_connector as db
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
load_dotenv()

def run_kmeans(k, pixels_2d, sample_size):
    kmeans = KMeans(n_clusters=k, random_state=22)
    labels = kmeans.fit_predict(pixels_2d)
    score = silhouette_score(pixels_2d, labels, sample_size=sample_size)  # Calcular silhouette coefficient
    return score

app = Flask(__name__)
app.secret_key = os.environ['secret_key']
app.config['UPLOAD_FOLDER'] = os.environ['upload_path']

nav_items = {}

def load_pages_and_register_routes(app: Flask, pages_folder: str = "pages"):
    """Carga las páginas de forma dinámica de la carpeta pages."""
    modules_path = os.path.join(os.path.dirname(__file__), pages_folder)
    
    for filename in os.listdir(modules_path):
        if filename.endswith(".py") and not filename.startswith("__"):
            module_name = filename[:-3]
            module_path = os.path.join(modules_path, filename)
            
            # Cargar páginas de forma dinámica
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Registrar páginas decoradas
            for attr_name in dir(module):
                if attr_name.startswith('__'):
                    continue

                attr = getattr(module, attr_name)
                if not inspect.isfunction(attr):
                    continue

                if hasattr(attr, "route") and hasattr(attr, "methods"):
                    app.route(attr.route, methods=attr.methods)(attr)
                
                access = ['public']
                if hasattr(attr, "restricted"):
                    access = attr.restricted

                if hasattr(attr, "nav"):
                    for person in access:
                        if person not in nav_items:
                            nav_items[person] = []

                        offline = hasattr(attr, "offline")
                        nav_items[person].append({
                            'text': attr.nav,
                            'url': attr.route,
                            'offline': offline
                        })

                        print(f"Registrado {attr.nav} como offline:{offline}.")

#Descomentar cuando ya quede todo por que no sé en qué parte del código estoy mal jajaja
@app.errorhandler(Exception)
def page_not_found(e):  
    if isinstance(e, HTTPException):
        return render_template('error.html', description=str(e), code=e.code), e.code
  
    return render_template('error.html', description=str(e), code=500), 500

# Load modules and register routes
load_pages_and_register_routes(app)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
