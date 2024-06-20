from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from Helper import load_in_db

text_splits = load_in_db()
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Model for creating vector embeddings
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever_vectordb = vectorstore.as_retriever(search_kwargs={"k": 5})

keyword_retriever = BM25Retriever.from_documents(text_splits)
keyword_retriever.k =  5

ensemble_retriever = EnsembleRetriever(retrievers=[retriever_vectordb,keyword_retriever],
                                       weights=[0.5, 0.5])

query="What is marketing mix modeling ?"

docs_rel=ensemble_retriever.get_relevant_documents(query)
print(docs_rel)