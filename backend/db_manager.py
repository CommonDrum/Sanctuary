# Init DB

# Load file

# Store config

#Use the chroma module separate from langchain
#https://docs.trychroma.com/docs/overview/getting-started


import chromadb
import os

class DBManager:
     def __init__(self, documents_path):
          self.client = chromadb.PersistentClient(path="db/chroma_db")
          self.note_collection = self.client.create_collection(name="notes")

          self.load_documents(documents_path)

     def semantic_search(self, query : list[str]):
          return self.note_collection.query(query_texts=query,
                                            n_results=10,
                                             )
     def text_search(self, query : str):
          return self.note_collection.query(query_text=query,
                                            n_results=10,
                                             )
     def load_documents(self, documents_path):
          for file in os.listdir(documents_path):
              with open(os.path.join(documents_path, file)) as f:
                    content = f.read()
                    document_id = hash(file)
                    self.note_collection.add(
                         documents=[content],
                         ids=[document_id]
                         )
     