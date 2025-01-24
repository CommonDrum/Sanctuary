from langchain_core.documents import Document
from langgraph.graph import START, END, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from dataclasses import dataclass

import requests

from template import SOCIAL_MEDIA_TEMPLATE

#TODO: Write docs!
@dataclass
class AgentConfig:
    model_name: str = "llama3.2:3b"
    base_url: str = "http://ollama:11434"
    chunk_size: int = 1000
    chunk_overlap: int = 200

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

class Agent:
     """
               args:
                    template: ChatPromptTemplate
                    model_name: str
               This 
     """
     def __init__(self, template, persist_dir="./db/chroma_db", model_name="llama3.2:3b"):

          self.presist_dir = persist_dir
          self.config = AgentConfig()          
          
          self.model_name = model_name
          self.embeddings = OllamaEmbeddings(model=self.model_name)
          self.llm = OllamaLLM(
               model=self.model_name,
               )
          
          self.vector_store = Chroma(embedding_function=self.embeddings, persist_directory=self.presist_dir) 
          self.prompt = template
          self.State = State
         
     def retrieve(self, state: TypedDict):
          retrieved_docs = self.vector_store.similarity_search(state["question"])
          return {"context": retrieved_docs}
     
     def generate(self, state: TypedDict):
          docs_content = "\n\n".join(doc.page_content for doc in state["context"])
          messages = self.prompt.invoke({"question": state["question"], "context": docs_content})
          response = self.llm.invoke(messages)
          return {"answer": response}
     
     def create_graph(self):
          graph_builder = StateGraph(self.State)
          
          # NODES
          graph_builder.add_node("retrieve", self.retrieve)
          graph_builder.add_node("generate", self.generate)

          # EGDGES
          graph_builder.add_edge(START, "retrieve")
          graph_builder.add_edge("retrieve", "generate")
          graph_builder.add_edge("generate", END)

          return graph_builder.compile()
     
     def ask_question(self, question: str):
          graph = self.create_graph()
          response = graph.invoke({"question": question})
          return response["answer"]



if __name__ == "__main__":
     import requests
     import json
     import time

     def pull_model():
          url = "http://ollama:11434/api/pull"
          payload = {
               "model": "llama3.2:3b"
          }
          
          print("Initializing model...")
          response = requests.post(url, json=payload)
          return response.status_code == 200

     def generate_text(prompt):
          url = "http://ollama:11434/api/generate"
          payload = {
               "model": "llama3.2:3b",
               "prompt": prompt
          }
          
          response = requests.post(url, json=payload)
          return response
     
     while True:
            try:
               if pull_model():
                    print(generate_text("Hello, world!"))
                    print("Model initialized!")
                    agent = Agent(SOCIAL_MEDIA_TEMPLATE)
                    print(agent.ask_question("Hello, world!"))
                    break
               else:
                    time.sleep(500)
            except requests.exceptions.ConnectionError:
               time.sleep(1)
        