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

class Agent:
    """
          args:
               template: ChatPromptTemplate
               model_name: str
          This a class for creating an agent that can answer questions based on a template
          The premise is to use it with local models and data
          
    """
    def __init__(self, template, model_name = "llama3.2:3b"):

        self.loader = TextLoader("projects.txt")
        self.docs = self.loader.load()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.all_splits = self.text_splitter.split_documents(self.docs)
        
        self.model_name = model_name
        self.embeddings = embeddings = OllamaEmbeddings(model=self.model_name)
        self.llm = Ollama(
          model=self.model_name,
          base_url="http://localhost:11434",
          )
        
        self.vector_store = Chroma(embedding_function=embeddings)
        self.vector_store.add_documents(documents=self.all_splits)
        self.prompt = template
        
        class State(TypedDict):
            question: str
            context: List[Document]
            answer: str
        
        self.State = State

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

