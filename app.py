import os
import oracledb
import google.genai as genai
from flask import Flask, request, render_template, jsonify
import pdfplumber

# =========================================================================
# Configuration
# =========================================================================

# Database connection details
ORACLE_USER = os.environ.get("ORACLE_USER", "sample")
ORACLE_PASSWORD = os.environ.get("ORACLE_PASSWORD", "XXXXXXXXXXXX")
ORACLE_DSN = os.environ.get("ORACLE_DSN", "localhost:1521/freepdb1")

# Gemini API key
GENAI_API_KEY = os.environ.get("GEMINI_API_KEY", "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

# Create a Flask web application
app = Flask(__name__)

# =========================================================================
# RAG Chatbot Implementation
# =========================================================================

def ora_chatbot_rag(user_question: str, num_chunks: int = 3) -> str:
    """
    RAG-based chatbot function that uses Oracle DB for context retrieval.
    """
    try:
        # Step 1: Connect to the database and get the user query's embedding
        print("Step 1: Embedding the user query...")
        with oracledb.connect(user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=ORACLE_DSN) as connection:
            with connection.cursor() as cursor:
                # Get the embedding vector for the user's question
                embedding_sql = """
                    SELECT VECTOR_EMBEDDING(MINILM_EMBEDDING_MODEL USING :user_data AS data) FROM DUAL
                """
                cursor.execute(embedding_sql, user_data=user_question)
                query_embedding = cursor.fetchone()[0]

        # Step 2: Perform the vector similarity search to retrieve relevant chunks
        print("Step 2: Retrieving relevant document chunks...")
        retrieved_chunks_text = []
        with oracledb.connect(user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=ORACLE_DSN) as connection:
            with connection.cursor() as cursor:
                # Use the query embedding to find the most similar document chunks
                retrieval_sql = f"""
                    SELECT embed_data
                    FROM ai_doc_chunks
                    ORDER BY VECTOR_DISTANCE(embed_vector, :query_embedding, COSINE)
                    FETCH FIRST {num_chunks} ROWS ONLY
                """
                cursor.execute(retrieval_sql, query_embedding=query_embedding)
                for row in cursor:
                    retrieved_chunks_text.append(row[0])

        if not retrieved_chunks_text:
            return "I'm sorry, I couldn't find any relevant documentation to answer your question."

        # Step 3: Combine context and question, then generate a response with an LLM
        print("Step 3: Generating a response with the LLM...")
        
        client = genai.Client(api_key=GENAI_API_KEY)
        
        # Create the enriched prompt
        context = "\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(retrieved_chunks_text)])
        prompt = f"""
            You are a helpful assistant for Oracle database administrators.
            Use the following documentation snippets to answer the user's question.
            If the snippets do not contain enough information, state that you cannot answer the question based on the provided context.

            Documentation Context:
            {context}

            User Question: {user_question}
            """

        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt
        )
        return response.text

    except oracledb.Error as e:
        print(f"Database error: {e}")
        return "An error occurred while connecting to or querying the database."
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return "An unexpected error occurred during the process."

# =========================================================================
# PDF Processing and Database Insertion
# =========================================================================

def process_and_insert_pdf(pdf_file) -> str:
    """
    Extracts text from a PDF, chunks it, and inserts it into the database.
    """
    try:
        # Connect to the database
        with oracledb.connect(user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=ORACLE_DSN) as connection:
            with connection.cursor() as cursor:
                # Use pdfplumber to extract text from the PDF file
                with pdfplumber.open(pdf_file) as pdf:
                    text_content = ""
                    for page in pdf.pages:
                        text_content += page.extract_text() or ""
                
                if not text_content:
                    return "Error: Could not extract text from the PDF."

                # Insert the full document text into the ai_documentation_tab
                insert_sql = """
                    INSERT INTO ai_documentation_tab (document_name, document_text)
                    VALUES (:doc_name, :doc_text)
                """
                cursor.execute(insert_sql, doc_name=pdf_file.filename, doc_text=text_content)
                connection.commit()
                print(f"Successfully inserted {pdf_file.filename} into ai_documentation_tab.")
                return "PDF uploaded and processed successfully."

    except oracledb.Error as e:
        print(f"Database error during PDF upload: {e}")
        return f"An error occurred while writing to the database: {e}"
    except Exception as e:
        print(f"An unexpected error occurred during PDF processing: {e}")
        return f"An unexpected error occurred: {e}"

# =========================================================================
# Flask Routes
# =========================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_query = data.get('query')
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    
    response = ora_chatbot_rag(user_query)
    return jsonify({"response": response})

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    """
    Handles the PDF file upload request.
    """
    if 'pdf_file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['pdf_file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith('.pdf'):
        # Process the uploaded file
        message = process_and_insert_pdf(file)
        return jsonify({"message": message})
    else:
        return jsonify({"error": "Invalid file type. Please upload a PDF file."}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)