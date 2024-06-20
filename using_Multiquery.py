from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import os
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
import pandas as pd
from langchain.retrievers.multi_query import MultiQueryRetriever


# Load the Google API key from the .env file
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    article_content = soup.find_all('p')
    text = '\n'.join([p.get_text() for p in article_content])  # Combine all paragraphs
    return text.encode('ascii', 'ignore').decode('ascii')  # Remove non-ASCII characters


def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as pdfFile:
        pdfReader = PdfReader(pdfFile)  # Used PdfReader from PyPDF2 to read the PDF
        numPages = len(pdfReader.pages)
        all_text = ""
        for page_num in range(numPages):
            page = pdfReader.pages[page_num]
            text = page.extract_text()
            if text:
                all_text += text.encode('ascii', 'ignore').decode('ascii') + "\n"  # Ignore Non-ASCII characters
    return all_text

def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    article_content = soup.find_all('p')
    text = '\n'.join([p.get_text() for p in article_content])  # Combine all paragraphs
    return text.encode('ascii', 'ignore').decode('ascii')  # Remove non-ASCII characters

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, batch_size=100):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Model for creating vector embeddings
    text_embeddings = []
    
    # Process text chunks in batches
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        batch_embeddings = embeddings.embed_documents(batch)  # Generate embeddings for the batch
        text_embeddings.extend(zip(batch, batch_embeddings))  # Pair text chunks with their embeddings
    # print(text_embeddings)
    vector_store = FAISS.from_embeddings(text_embeddings, embedding=embeddings)  # Pass the embedding model here
    
    vector_store.save_local("faiss_index")  # Save the vector database
    return vector_store

def get_conversational_chain():
    prompt_template = """
    Try to understand the context and then give detailed answers as much as possible. Don't answer if answer is not from the context.
    provide every answer in detailed explanation and easy words to make easy for the User.
    Also, provide one URL link given in the context in the following way in the end of the Answer.
    "For more details visit" : URL link \n\n
    Context:\n{context}?\n
    Question:\n{question} + Explain in detail.\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
    # model = GPT4All(model_name="gpt4all-lora-quantized")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain , model

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Model for creating vector embeddings
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)  # Load the previously saved vector db
    chain , model = get_conversational_chain()

    mq_retriever = MultiQueryRetriever.from_llm(retriever = new_db.as_retriever(k = 6), llm = model)
    docs = mq_retriever.get_relevant_documents(query=user_question)


    # docs = new_db.similarity_search(query=user_question, k=10)  # Get similar text from the database with the query
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response, docs

def load_in_db():
    file_path = 'Article_Links.xlsx'  # Update this with the actual path to your Excel file
    df = pd.read_excel(file_path, header=None)
    url_text_chunks = []

    for url in df[0]:
        article_text = extract_text_from_url(url)
        text_chunks = get_text_chunks(article_text)
        for chunk in text_chunks:
            url_text_chunks.append(f"URL: {url}\n{chunk}")
    # text_chunks = get_text_chunks(all_text)
    get_vector_store(url_text_chunks)
    return url_text_chunks

def main():
    load_in_db()

# def main():
#     all_text = ""
#     file_path = 'Article_Links.xlsx'  # Update this with the actual path to your Excel file
#     df = pd.read_excel(file_path, header=None)
#     url_text_chunks = []

#     for url in df[0]:
#         article_text = extract_text_from_url(url)
#         text_chunks = get_text_chunks(article_text)
#         for chunk in text_chunks:
#             url_text_chunks.append(f"URL: {url}\n{chunk}")

#     # Example: Print all chunks with their corresponding URLs
#     for url_chunk in url_text_chunks:
#         print(url_chunk)
#         print("\n---\n")

if __name__ == "__main__":
    main()

