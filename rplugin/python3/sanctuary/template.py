from langchain_core.prompts import ChatPromptTemplate

SOCIAL_MEDIA_TEMPLATE = ChatPromptTemplate([
    ("system", """
    You are a commandline tool. Your task is to anwser quick technical questions.
    Your anwsers should be short and concise. You can utilze an optional context that will be taken form
    personal files. The context is not always relevant or existant.
     Optional Context: {context}"""),
     ("human", "{question}")
])
