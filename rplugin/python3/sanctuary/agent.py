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
from enum import Enum


"""
TODO:
- add chat history
- add deepseek init for the class
- add config file
- Loading documentation from URL
"""
@dataclass
class AgentConfig:
    """Configuration settings for the Agent.
    
    Attributes:
        model_name: Name of the LLM model to use. Defaults to "llama3.2:3b".
        base_url: Base URL for the Ollama server. Defaults to "http://localhost:11434".
        chunk_size: Size of text chunks for document processing. Defaults to 1000.
        chunk_overlap: Overlap between text chunks. Defaults to 200.
    """
    model_name: str = "llama3.2:3b"
    base_url: str = "http://localhost:11434"
    chunk_size: int = 1000
    chunk_overlap: int = 200

class State(TypedDict):
    """State representation for the agent workflow.
    
    Attributes:
        user_input: User's input question or prompt.
        context: List of relevant documents retrieved for context.
        answer: Generated answer from the LLM.
    """
    user_input: str
    context: List[Document]
    answer: str

class FileType(Enum):
    """Enumeration of supported file types and their loaders.
    
    Members:
        MD: Markdown files (.md) with TextLoader
        PY: Python files (.py) with PythonLoader
    """
    MD = (".md", TextLoader)
    PY = (".py", PythonLoader)

class Agent:
    """Main agent class for handling document retrieval and generation workflows.
    
    Args:
        template: Chat prompt template to use. Defaults to SOCIAL_MEDIA_TEMPLATE.
        model_name: LLM model name. Defaults to "llama3.2:3b".
        host_url: (Currently unused) Base URL for model hosting service.

    Attributes:
        config: Agent configuration settings
        embeddings: Text embeddings model
        llm: Large Language Model interface
        vector_store: In-memory vector store for documents
        prompt: Chat prompt template
        State: Type reference for workflow state
    """

    def __init__(self, template=SOCIAL_MEDIA_TEMPLATE, model_name="llama3.2:3b", host_url="http://localhost:11434"):
        self.config = AgentConfig()          
        self.config.model_name = model_name
        self.embeddings = OllamaEmbeddings(model=self.config.model_name)
        # TODO: Implement host URL configuration
        self.llm = OllamaLLM(model=self.config.model_name)
        self.vector_store = InMemoryVectorStore(self.embeddings) 
        self.prompt = template
        self.State = State

    def retrieve(self, state: State) -> dict:
        """Retrieve relevant documents based on user input.
        
        Args:
            state: Current workflow state containing user input

        Returns:
            Dictionary with "context" key containing retrieved documents
        """
        retrieved_docs = self.vector_store.similarity_search(state["user_input"])
        return {"context": retrieved_docs}

    def generate(self, state: State) -> dict:
        """Generate answer using LLM with context documents.
        
        Args:
            state: Current workflow state containing user input and context

        Returns:
            Dictionary with "answer" key containing generated response
        """
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke({"user_input": state["user_input"], "context": docs_content})
        response = self.llm.invoke(messages)
        return {"answer": response}

    def create_graph(self) -> StateGraph:
        """Create and configure the workflow state graph.
        
        Returns:
            Compiled state graph with nodes and edges
        """
        graph_builder = StateGraph(self.State)
        graph_builder.add_node("retrieve", self.retrieve)
        graph_builder.add_node("generate", self.generate)
        graph_builder.add_edge(START, "retrieve")
        graph_builder.add_edge("retrieve", "generate")
        graph_builder.add_edge("generate", END)
        return graph_builder.compile()

    def ask_question(self, question: str) -> str:
        """Execute full workflow to answer a question.
        
        Args:
            question: User's input question to answer

        Returns:
            Generated answer string
        """
        graph = self.create_graph()
        response = graph.invoke({"user_input": question})
        return response["answer"]

    def swap_to_deepseek(self, model_name: str) -> None:
        """Switch LLM to DeepSeek model.
        
        Args:
            model_name: DeepSeek model name to use

        Raises:
            EnvironmentError: If DEEPSEEK_API_KEY is not set
        """
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise EnvironmentError("DEEPSEEK_API_KEY not found in environment variables")
        
        self.config.model_name = model_name
        self.llm = ChatDeepSeek(
            model=self.config.model_name,
            temperature=0.3,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

    def swap_to_ollama(self, model_name: str) -> None:
        """Switch LLM to Ollama model.
        
        Args:
            model_name: Ollama model name to use
        """
        self.config.model_name = model_name
        self.llm = OllamaLLM(model=self.config.model_name)

    def load_dir_to_vector(self, dir: str, filetype: FileType) -> None:
        """Load documents from directory into vector store.
        
        Args:
            dir: Directory path to load documents from
            filetype: FileType enum specifying which files to load
        """
        extension, loader_class = filetype.value
        loader = DirectoryLoader(
            dir,
            glob=f"**/*{extension}",
            loader_cls=loader_class,
        )
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        all_splits = text_splitter.split_documents(docs)
        self.vector_store.add_documents(documents=all_splits)

    def clear_vector_memory(self) -> None:
        """Reset the vector store to empty state."""
        self.vector_store = InMemoryVectorStore(self.embeddings)


if __name__ == "__main__":
    agent = Agent(SOCIAL_MEDIA_TEMPLATE)
    response = agent.ask_question("how to print in python?")
    print(response)
