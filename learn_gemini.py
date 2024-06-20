from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
# setup the gemini pro
from dotenv import load_dotenv
load_dotenv()
os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)
conversation.predict(input="Hi there!")
