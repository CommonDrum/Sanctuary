from langchain_core.documents import Document
from langgraph.graph import START, END, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import DirectoryLoader, PythonLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dataclasses import dataclass
import os
from .template import SOCIAL_MEDIA_TEMPLATE
from langchain_deepseek import ChatDeepSeek


"""
TODO:

- add chat history
- create documentation
- add model swap feature
- make markdown loader work
- add deepseek init for the class
- add config file

"""
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
   
    def __init__(self, template=SOCIAL_MEDIA_TEMPLATE, model_name="llama3.2:3b", host_url="http://localhost:11434"):

            self.config = AgentConfig()          
            self.config.model_name = model_name
            self.embeddings = OllamaEmbeddings(model=self.config.model_name)
            #TODO: Check how to change URL for hosting
            self.llm = OllamaLLM(
                model=self.config.model_name,
            )
            self.conversation = Null 
            self.vector_store = InMemoryVectorStore(self.embeddings) 
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

            graph_builder.add_node("retrieve", self.retrieve)
            graph_builder.add_node("generate", self.generate)

            graph_builder.add_edge(START, "retrieve")
            graph_builder.add_edge("retrieve", "generate")
            graph_builder.add_edge("generate", END)

            return graph_builder.compile()

        def ask_question(self, question: str):
            graph = self.create_graph()
            response = graph.invoke({"question": question})
            return response["answer"]

        def swap_to_deepseek(self, model_name):

            if not os.getenv("DEEPSEEK_API_KEY"):
                print("Add DEEPSEEK_API_KEY to your environment variables")

            self.config.model_name = model_name

            self.llm = ChatDeepSeek(
                model=config.model_name,
                temperature=0.3,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )


        def swap_to_ollama(self, model_name):

            self.config.model_name = model_name

            self.llm = OllamaLLM(
                model=self.config.model_name,
            )


if __name__ == "__main__":
    agent = Agent(SOCIAL_MEDIA_TEMPLATE)
    response = agent.ask_question("What is agent class?")
    print(response)
