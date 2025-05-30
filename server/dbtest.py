from datetime import datetime
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import bson
import os

uri = os.environ['uri']

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Get a reference to the 'sample_mflix' database:
db = client['captacion']

# List all the collections in 'sample_mflix':
collections = db.list_collection_names()
for collection in collections:
   print(collection)

formulario = db['formulario']

# Insert a document for the movie 'Parasite':
insert_result = formulario.insert_one({
      "curp": "Adios",
      "nombre": "Juancho",
      "hermano": bson.ObjectId("673e98eeaac2ca36d7a41cb2")
   })

# Save the inserted_id of the document you just created:
inserted_id = insert_result.inserted_id
print("_id of inserted document: {parasite_id}".format(parasite_id=inserted_id))