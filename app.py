import request
import numpy as np
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import spacy

# Initialize Flask app and API
app = Flask(__name__)
api = Api(app)

# Global variable to hold the vector store
vector_store = faiss.read_index("my_index.faiss")
model = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_sm")
sentences = pickle.load(open("sentences.pkl", "rb"))

def get_response(query, k=2):
    query_embedding = model.encode([query]).astype('float32')  # Encode the query to embedding
    D, I = vector_store.search(query_embedding, k)  # Perform similarity search using FAISS
    return [{'sentence': sentences[I[0][i]], 'distance': float(D[0][i])} for i in range(k)]

class Chatbot(Resource):
    def post(self):
        query = request.json.get("query")  # Get the query from JSON body
        k = request.json.get("k", 3)  # Get the query from JSON body
        if not query:
            return {"error": "Query is required"}, 400
        
        # Call get_response function to get the top-k responses
        result = get_response(query, k)
        
        # Return the response in JSON format
        return jsonify(result)

# Add resources to the API
api.add_resource(Chatbot, '/chatbot')

if __name__ == '__main__':
    app.run(debug=True)
