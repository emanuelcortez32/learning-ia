from langchain_ollama import ChatOllama
from models import AgentChat
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

SYSTEM_PROMPT = """
    You are a friendly and expert Wikipedia Searcher
"""

LLM_MODEL = "llama3.2:3b"

model = ChatOllama(
    model=LLM_MODEL,
    base_url="http://localhost:11434",
    temperature=0.3
)

wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

wikipedia_agent = AgentChat(model=model, system_prompt=SYSTEM_PROMPT, tools=[wikipedia_tool])