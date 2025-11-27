# import libraries
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
import os
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv() 

class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]

groq_key = os.environ["GROQ_API_KEY"]

# Create model
model = ChatGroq(
    api_key=groq_key,
    model="llama-3.1-8b-instant"
)

# Checkpointer
checkpointer = InMemorySaver()

def chat_node(state: ChatState):

    # take user query from state
    messages = state['messages']

    # send to llm
    response = model.invoke(messages)

    # response store state
    return {'messages': [response]}

graph = StateGraph(ChatState)

# add nodes
graph.add_node('chat_node', chat_node)

# add edges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

# Compile the graph
chatbot = graph.compile(checkpointer=checkpointer)