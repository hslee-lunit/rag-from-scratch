from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank


class DocumentRetriever:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever()
        self.compressor = CohereRerank()
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor, base_retriever=self.retriever
        )

    def retrieve(self, query, k=5):
        return self.compression_retriever.get_relevant_documents(query)[:k]
