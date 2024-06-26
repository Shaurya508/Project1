from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import os
# from PyPDF2 import PdfReader #used it before now using tesseract
import requests
from bs4 import BeautifulSoup
# from dotenv import load_dotenv
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
import pandas as pd
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain.chains import ConversationChain
from pdf2image import convert_from_path
# from PIL import Image
import streamlit as st
import pytesseract
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.common.keys import Keys
# import time
# Load the Google API key from the .env file
# load_dotenv()
# genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
# Load the Google API key from the .env file
# load_dotenv()
# genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
# genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# sec_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")
# os.environ["HUGGINGFACEHUB_API_TOKEN"]=sec_key
# Function to log in to LinkedIn
def linkedin_login(email, password , driver):
    driver.get("https://www.linkedin.com/login")
    
    # Find the username/email field and send the email
    email_field = driver.find_element(By.ID, "username")
    email_field.send_keys(email)
    
    # Find the password field and send the password
    password_field = driver.find_element(By.ID, "password")
    password_field.send_keys(password)
    
    # Submit the form
    password_field.send_keys(Keys.RETURN)
    
    # Wait for a bit to allow login to complete
    time.sleep(5)

# Function to scrape LinkedIn post content
def scrape_linkedin_post(url , driver):
    # Open the LinkedIn post URL
    driver.get(url)
    
    # Wait for the content to load
    time.sleep(5)
    
    # Find the main content of the post
    # Note: The actual class names and structure may vary, so you might need to inspect the LinkedIn post's HTML to get the accurate class names or ids
    post_content = driver.find_element(By.CLASS_NAME, 'feed-shared-update-v2__description')
    
    # Extract and return the text content
    if post_content:
        return post_content.text.encode('ascii', 'ignore').decode('ascii')
    else:
        return "Could not find the main content of the post."



def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    article_content = soup.find_all('p')
    text = '\n'.join([p.get_text() for p in article_content])
    return text.encode('ascii', 'ignore').decode('ascii')

def extract_text_from_pdf(pdf_path):
    # Convert PDF to images
    pages = convert_from_path(pdf_path, 300)

    # Iterate through all the pages and extract text
    extracted_text = ''
    for page_number, page_data in enumerate(pages):
        # Perform OCR on the image
        text = pytesseract.image_to_string(page_data)
        extracted_text += f"Page {page_number + 1}:\n{text}\n"
    return extracted_text
#     with open(pdf_path, "rb") as pdfFile:
#         pdfReader = PdfReader(pdfFile)
#         numPages = len(pdfReader.pages)
#         all_text = ""
#         for page_num in range(numPages):
#             page = pdfReader.pages[page_num]
#             text = page.extract_text()
#             if text:
#                 all_text += text.encode('ascii', 'ignore').decode('ascii') + "\n"
#     return all_text

def extract_code_from_github(raw_url):
    # text = extract_text_from_url('https://github.com/facebookexperimental/Robyn/blob/main/demo/demo.R')
    # print(text)
    # URL of the raw content of the R script on GitHub
    # raw_url = "https://raw.githubusercontent.com/facebookexperimental/Robyn/main/demo/demo.R"

    # Send a GET request to the raw content URL
    response = requests.get(raw_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Get the content of the response
        code = response.text

        # Print the scraped code
        return code
    # else:
    #     print(f"Failed to retrieve the URL. Status code: {response.status_code}")

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks, batch_size=100):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    text_embeddings = []
    
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        batch_embeddings = embeddings.embed_documents(batch)
        text_embeddings.extend(zip(batch, batch_embeddings))
    
    vector_store = FAISS.from_embeddings(text_embeddings, embedding=embeddings)
    vector_store.save_local("faiss_index2")
    return vector_store


# Initialize an empty list to store conversation history
conversation_history = []

def user_input(user_question):
    # global conversation_history  # Access the global conversation history list
    
    # # Initialize the model and embeddings
    # model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # # Load the FAISS vector store
    # vector_db = FAISS.load_local("faiss_index1", embeddings, allow_dangerous_deserialization=True)
    
    # # Initialize the retriever with the vector store
    # retriever = MultiQueryRetriever.from_llm(retriever=vector_db.as_retriever(k=8), llm=model)
    # docs = retriever.get_relevant_documents(query=user_question)
    # Retrieve relevant documents based on the current question only
    # input_documents = retriever.get_relevant_documents(query=user_question)
    
#     prompt_template = """ You are a chatbot named MMM GPT developed by ARYMA LABS having a conversation with a human.Try to understand the context, chat history and question properly and then give detailed answers as much as possible. Don't answer if answer is not from the context however do talk like a human in a well mannered way.
#     provide every answer in detailed explanation and easy words to make easy for the User.Give more weightage to question.
#     when the question is asking for code:
#     provide Short part of code Write that "Code looks like this " before giving code in a good format and easy to copy format.after providing the code , give Github Link from the paragraph from which you give the code in the end like :
#     "Please check detailed code at" : Github link
#     when question is not for code :
#     but always provide link!
#     Always provide one URL link or linkidin blog link given in the context in the following way in the end of the Answer whenver you think it's correct to provide link.
#     "For more details visit" : URL link or linkidin link \n\n

#     chat history : {chat_history}
#     context : {context}
#     question: {human_input}
# Chatbot:"""
    prompt_template = """
    you are MMMGPT devloped by Aryma labs to help users on market mix modelling(MMM) , Now you have to chat with the user.
    the context given is from more important to less important from top to bottom.
    Try to understand the context and then give detailed answers as much as possible. Don't answer if answer is not from the context.
    provide every answer in detailed explanation and easy words to make easy for the User.
    Also, provide one URL link given in the context only in the following way in the end of the Answer.
    "For more details visit" : link \n\n
    Always give link associated with the context and don't provide any link if there is no link in the first five paragraph of the context.
    Context:\n{context}?\n
    Question:\n{question} + Explain in detail.\n
    Answer:
    """
#     PROMPT = PromptTemplate(
#         template=prompt_template, input_variables=["context" , "question"]
#     )
#     # chain_type_kwargs = { "prompt" : PROMPT }

#     memory = ConversationBufferMemory(memory_key='chat_history',input_key="human_input" , return_messages=True)
#     # conversation_chain = ConversationalRetrievalChain.from_llm(
#     #     llm = model,
#     #     retriever= vector_db.as_retriever(k=6),
#     #     memory = memory,
#     #     combine_docs_chain_kwargs=chain_type_kwargs
#     # )
#     chain = load_qa_chain(
#     model, chain_type="stuff", memory=memory, prompt=PROMPT
# )
#     return chain({"input_documents": docs, "human_input": user_question}, return_only_outputs=True) , docs
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.9)
    # repo_id="mistralai/Mistral-7B-Instruct-v0.2"
    # model=HuggingFaceEndpoint(repo_id=repo_id,max_length=128,temperature=0.7,token=sec_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Model for creating vector embeddings
    new_db = FAISS.load_local("faiss_index3", embeddings, allow_dangerous_deserialization=True)  # Load the previously saved vector db
    # chain , model = get_conversational_chain()

    mq_retriever = MultiQueryRetriever.from_llm(retriever = new_db.as_retriever(k = 8), llm = model)
    docs = mq_retriever.get_relevant_documents(query=user_question)


    # docs = new_db.similarity_search(query=user_question, k=10)  # Get similar text from the database with the query
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response, docs

    # response  = conversation_chain({"question": user_question})
    # return(response.get("answer"))


def load_in_db():
    file_path = 'Article_Links.xlsx'
    df = pd.read_excel(file_path, header=None)
    url_text_chunks = []

    for url in df[0]:
        article_text = extract_text_from_url(url)
        text_chunks = get_text_chunks(article_text)
        for chunk in text_chunks:
            url_text_chunks.append(f"URL: {url}\n{chunk}")
    pdf_titles = ['Robyn.pdf' , 'Robyn-Under-The-Hood.pdf' , 'Robyn-Under-the-hood-Ridge-Regression-in-depth.pdf' , '5.-Robyn-Under-The-Hood-Flowchart_v2.pdf', '3.-Robyn-Under-The-Hood-Adstock-Transformation.pdf', '4.-Robyn-Under-The-Hood-Hill-Transformation.pdf','Granger-Causality-A-possible-Feature-Selection-Method-in-Marketing-Mix-Modeling-MMM.pdf','Granularity-MMM-Problem-Cracked.pdf' , 'Investigation-of-Marketing-Mix-Models-Business-Error-using-KL-Divergence-and-Chebyshevs-Inequality.pdf', 'Proving-Efficacy-of-Marketing-Mix-Model-through-the-Difference-in-Difference-DID-Technique.pdf']
    for pdf in pdf_titles:
        article_text = extract_text_from_pdf(pdf)
        text_chunks = get_text_chunks(article_text)
        for chunk in text_chunks:
            url_text_chunks.append(f"{chunk}")
    # Github stuff
    Github_links =  ['https://github.com/facebookexperimental/Robyn/blob/main/demo/demo.R' ,'https://github.com/facebookexperimental/Robyn/blob/main/R/R/allocator.R' ,'https://github.com/facebookexperimental/Robyn/blob/main/R/R/auxiliary.R' ,'https://github.com/facebookexperimental/Robyn/blob/main/R/R/calibration.R' , 'https://github.com/facebookexperimental/Robyn/blob/main/R/R/checks.R' ,'https://github.com/facebookexperimental/Robyn/blob/main/R/R/clusters.R' , 'https://github.com/facebookexperimental/Robyn/blob/main/R/R/convergence.R' , 'https://github.com/facebookexperimental/Robyn/blob/main/R/R/data.R' ,'https://github.com/facebookexperimental/Robyn/blob/main/R/R/exports.R' ,'https://github.com/facebookexperimental/Robyn/blob/main/R/R/imports.R' ,'https://github.com/facebookexperimental/Robyn/blob/main/R/R/inputs.R' ,'https://github.com/facebookexperimental/Robyn/blob/main/R/R/json.R' ,'https://github.com/facebookexperimental/Robyn/blob/main/R/R/model.R' ,'https://github.com/facebookexperimental/Robyn/blob/main/R/R/model.R' , 'https://github.com/facebookexperimental/Robyn/blob/main/R/R/outputs.R' ,'https://github.com/facebookexperimental/Robyn/blob/main/R/R/pareto.R' ,'https://github.com/facebookexperimental/Robyn/blob/main/R/R/plots.R' ,'https://github.com/facebookexperimental/Robyn/blob/main/R/R/refresh.R','https://github.com/facebookexperimental/Robyn/blob/main/R/R/response.R' , 'https://github.com/facebookexperimental/Robyn/blob/main/R/R/transformation.R' ,'https://github.com/facebookexperimental/Robyn/blob/main/R/R/zzz.R']
    for url in Github_links :
        article_text = extract_code_from_github(url)
        text_chunks = get_text_chunks(article_text)
        for chunk in text_chunks:
            url_text_chunks.append(f"Github Link : {url}\n{chunk}")
        # Linkidin blogs stuff

    # Set up the WebDriver (make sure chromedriver is in your PATH or provide the path to the executable)
    driver = webdriver.Chrome()

    linkedin_email = "shaurya@arymalabs.com"
    linkedin_password = "Mishra@123"
    # Log in to LinkedIn
    linkedin_login(linkedin_email, linkedin_password , driver)
    # linkedin_post_url = "https://www.linkedin.com/posts/ridhima-kumar7_marketingmixmodeling-marketingattribution-activity-7125811575931760640-Sx65?utm_source=share&utm_medium=member_desktop"

    file_path = 'Linkidin_blogs.xlsx'
    df = pd.read_excel(file_path, header=None)
    for linkedin_post_url in df[0]:
        post_text = scrape_linkedin_post(linkedin_post_url , driver)
        text_chunks = get_text_chunks(post_text)
        for chunk in text_chunks:
            url_text_chunks.append(f"Linkedin Link : {linkedin_post_url}\n{chunk}")

    get_vector_store(url_text_chunks)

def main():
    load_in_db()

if __name__ == "__main__":
    main()
