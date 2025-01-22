from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import Ollama

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
               This 
     """
     def __init__(self, template, persist_dir="./chroma_db", model_name="llama3.2:3b"):

          self.presist_dir = persist_dir
          self.config = AgentConfig()          
          
          self.model_name = model_name
          self.embeddings = OllamaEmbeddings(model=self.model_name)
          self.llm = Ollama(
               model=self.model_name,
               base_url="http://localhost:11434",
               )
          
          self.vector_store = Chroma(embeddings=self.embeddings, persist_dir=self.presist_dir) 
          self.prompt = template
          self.State = State()
          
     def retrieve(self, state: TypedDict):
          retrieved_docs = self.vector_store.similarity_search(state["question"])
          return {"context": retrieved_docs}
     
     def generate(self, state: TypedDict):
          docs_content = "\n\n".join(doc.page_content for doc in state["context"])
          messages = self.prompt.invoke({"question": state["question"], "context": docs_content})
          response = self.llm.invoke(messages)
          return {"answer": response.content}
     
     def create_graph(self):
          graph_builder = StateGraph(self.State).add_sequence([self.retrieve, self.generate])
          graph_builder.add_edge(START, "retrieve")
          return graph_builder.compile()
     
     def get_response(self, question: str):
          graph = self.create_graph()
          response = graph.invoke({"question": question})
          return response["answer"]

