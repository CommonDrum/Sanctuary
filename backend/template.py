from langchain_core.prompts import ChatPromptTemplate

SOCIAL_MEDIA_TEMPLATE = ChatPromptTemplate([
    ("system", """
     Optional Context: {context}
     Question: {question}
     Anwser: """)
])