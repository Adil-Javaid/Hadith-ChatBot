# import libraries
import pandas as pd
import os

# # List to store dataframes
# dfs = []

# # Loop through the files Chapter1 to Chapter43
# for i in range(1, 44):
#     # Load each chapter CSV file
#     filename = f'Chapter{i}.csv'
#     if os.path.exists(filename):
#         df = pd.read_csv(filename)
#         dfs.append(df)
#     else:
#         print(f"File {filename} does not exist.")

# # Concatenate all dataframes into one
# all_chapters_df = pd.concat(dfs, ignore_index=True)

# # Save the combined dataframe to a new CSV file
# all_chapters_df.to_csv('allChapters.csv', index=False)

# print("All chapters have been combined into 'allChapters.csv'.")

# Load the CSV file
# df = pd.read_csv('allChapters.csv')

# df

# df.columns

# # Select only the desired columns
# selected_columns = df[['Chapter_Number', 'Section_Number', 'Hadith_number', 'English_Hadith']]

# # Save the selected columns to a new CSV file
# selected_columns.to_csv('english_Hadith.csv', index=False)

# print("Selected columns have been saved to 'english_Hadith.csv'.") 

# df = pd.read_csv("english_Hadith.csv")
# df

from langchain_community.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(file_path="./english_Hadith.csv", encoding='utf-8')
data = loader.load()

# data

# data[0].page_content

# data[0].metadata


# Function to extract and move specific fields to metadata
def move_fields_to_metadata(doc):
    # Split the page_content into lines
    lines = doc.page_content.split('\n')
    
    # Initialize a dictionary to hold the new metadata
    new_metadata = {}
    
    # Initialize an empty list for new page content
    new_page_content = []
    
    # Iterate through each line in page_content
    for line in lines:
        if line.startswith('Chapter_Number:') or line.startswith('Section_Number:') or line.startswith('Hadith_number:'):
            # Split the line into key-value pair
            key, value = line.split(': ', 1)
            # Add key-value pair to the new metadata dictionary
            new_metadata[key] = value
        else:
            # Add non-metadata lines to the new page content
            new_page_content.append(line)
    
    # Update the document's metadata with the new metadata
    doc.metadata.update(new_metadata)
    
    # Update the document's page_content with the new content
    doc.page_content = '\n'.join(new_page_content)
    
    return doc

# Apply the function to each document
data = [move_fields_to_metadata(doc) for doc in data]

# The `data` list now contains documents with AyaID, SuraID, and AyaNo moved to metadata.


# data

# data[0].page_content
# print(data[0].metadata['Chapter_Number'], '-', data[0].metadata['Section_Number'], '-', data[0].metadata['Hadith_number'])

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(data, embedding_function)

# Persist the vectors locally on disk
vectorstore.save_local("Abu_Duad")

vectorstore = None

# Load from local storage
vectordb = FAISS.load_local("Abu_Duad", embedding_function, allow_dangerous_deserialization=True )
vectordb
query = "how long will be day of judgement"
docs = vectordb.similarity_search(query)
docs
for doc in docs:
    print(doc.page_content)
    print(doc.metadata)

results = vectordb.similarity_search(
    "result of disbelive",
    k=4,
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")

results = vectordb.similarity_search_with_score(
    "Who are successfull", k=10
)
for res, score in results:
    print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")

retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 5})
retriever.invoke("Who are successfull")

import random

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

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # This will render Hadith.html from the 'templates' directory

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

