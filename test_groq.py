import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model=os.environ["GROQ_MODEL"],
    temperature=0,
)

response = llm.invoke(
    "Explain RAG in simple terms with one example."
)

print(response.content)