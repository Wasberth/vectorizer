from wsgiref.simple_server import make_server
from main import app

# def application(environ, start_response):
#     headers = [('Content-type', 'text/plain; charset=utf-8')]

#     start_response('200 OK', headers)

#     return ['Hola gente de códigofacilito'.encode('utf-8')]

server = make_server('0.0.0.0', 5000, app)
print("Jalando en el puerto 5000")
server.serve_forever()