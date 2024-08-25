from flask import Flask, request, jsonify, render_template
import random
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

# Initialize Flask app
app = Flask(__name__)

# Load vectorstore from local storage
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = FAISS.load_local("Abu_Duad", embedding_function, allow_dangerous_deserialization=True)

def query_hadith_bot(query, top_k=1):
    # Clear the previous response
    response = []
    
    # Retrieve the most relevant documents based on the query
    results = vectordb.similarity_search(query, k=top_k * 2)  # Retrieve more than needed to ensure top_k results
    
    # Shuffle the results to introduce randomness
    random.shuffle(results)
    
    # Select the first 'top_k' results after shuffling
    for res in results[:top_k]:
        response.append({
            "Hadith": res.page_content.strip(),
            "Metadata": res.metadata
        })
    
    return response

@app.route('/')
def home():
    return render_template('index.html')  # Ensure 'index.html' exists in the 'templates' directory

@app.route('/query', methods=['POST'])
def query_hadith():
    user_query = request.json.get('query')
    
    if not user_query:
        app.logger.error("No query provided")
        return jsonify({"error": "No query provided"}), 400
    
    app.logger.info(f"Received query: {user_query}")
    
    response = query_hadith_bot(user_query)
    
    if not response:
        app.logger.error("No response from the query_hadith_bot")
        return jsonify({"error": "No response generated"}), 500
    
    app.logger.info(f"Generated response: {response}")
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
