from langchain_core.documents import Document
from langgraph.graph import START, END, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import DirectoryLoader, PythonLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dataclasses import dataclass
import os
from template import SOCIAL_MEDIA_TEMPLATE

#TODO: Write docs!
@dataclass
class AgentConfig:
    model_name: str = "llama3.2:3b"
    base_url: str = "http://localhost:11434"
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
                    host_url: str
     """
     def __init__(self, template, model_name="llama3.2:3b", host_url="http://localhost:11434"):

          self.config = AgentConfig()          
          
          self.model_name = model_name
          self.embeddings = OllamaEmbeddings(model=self.model_name)
          #TODO: Check how to change URL for hosting
          self.llm = OllamaLLM(
               model=self.model_name,
               )
          
          self.vector_store = InMemoryVectorStore(self.embeddings) 
          self.prompt = template
          self.State = State

          docs = self.load_directory("./rplugin/python/")

          text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
          all_splits = text_splitter.split_documents(docs)

          _ = self.vector_store.add_documents(documents=all_splits)

          #This loads ALL files form dir
          #TODO: make filtered version
     def load_directory(self, dir: str):
          # Verify path
          abs_path = os.path.abspath(dir)
          print(f"Loading from: {abs_path}")
          print(f"Directory exists: {os.path.exists(abs_path)}")
          print(f"Files found: {[f for f in os.listdir(abs_path) if f.endswith('.py')]}")
          
          loader = DirectoryLoader(
               dir,
               glob="**/*.py",
               exclude=["**/__pycache__"],
               use_multithreading=True,
               show_progress=True,
               silent_errors=True,
               loader_cls=PythonLoader
          )
          return loader.load()
               
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
     agent = Agent(SOCIAL_MEDIA_TEMPLATE)
     response = agent.ask_question("What is agent class?")
     print(response)
