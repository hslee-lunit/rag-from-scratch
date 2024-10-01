from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DataProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def process_documents(self, documents):
        processed_docs = []
        for doc in documents:
            clean_text = self._clean_html(doc.page_content)
            splits = self.text_splitter.split_text(clean_text)
            for i, chunk in enumerate(splits):
                processed_docs.append(Document(page_content=chunk, metadata={**doc.metadata, "chunk": i}))
        return processed_docs

    def _clean_html(self, html_content):
        soup = BeautifulSoup(html_content, "html.parser")
        text = soup.get_text(separator=" ")
        return text.strip()
