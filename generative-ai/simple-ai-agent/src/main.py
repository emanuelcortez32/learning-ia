from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

prompt = ChatPromptTemplate.from_messages([
    "system", "You are a helpful assistant. Answer the user's question concisely.",
    "human", "{question}"
])

model = "llama3.1:8b"
llm = ChatOllama(
    model=model,
    base_url="http://localhost:11434",
    temperature=0.3
)

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

response = chain.invoke({"question": "Dime que es la velocidad"})
print(response)