from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


class VectorStoreManager:
    def __init__(self, persist_directory="data/vectorstore"):
        self.persist_directory = persist_directory
        self.embedding_model = OpenAIEmbeddings()
        self.vectorstore = None

    def create_vectorstore(self, documents):
        self.vectorstore = Chroma.from_documents(
            documents, embedding=self.embedding_model, persist_directory=self.persist_directory
        )
        self.vectorstore.persist()

    def load_vectorstore(self):
        self.vectorstore = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding_model)
