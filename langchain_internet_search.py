"""
- dotenv used to retrieve OPENAI_API_KEY from .env file
Prompt:
What are autonomous agents and why they are popular in the world of AI right now?
"""
from dotenv import load_dotenv
import gradio as gr
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import DuckDuckGoSearchTool


load_dotenv()


def configure_tools():
    search = DuckDuckGoSearchTool()
    return [
        Tool(
            name="DuckDuckGo Search",
            func=search.run,
            description="Search the Internet if you don't know the answer.",
        ),
    ]


def query(input_text):
    llm = ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo")
    memory = ConversationBufferMemory(return_messages=True)
    tools = configure_tools()
    agent_chain = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
    )
    response = agent_chain.run(input=input_text)
    return response


interface = gr.Interface(
    fn=query,
    inputs="text",
    outputs="text",
    description="Your virtual assistant",
)

if __name__ == "__main__":
    interface.launch(share=True)
