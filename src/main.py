import os

from dotenv import load_dotenv

from data_loader.confluence_loader import ConfluenceLoader
from data_processor.processor import DataProcessor
from llm.llm_manager import LLMManager
from retriever.retriever import DocumentRetriever
from vectorstore.vectorstore_manager import VectorStoreManager

load_dotenv()


def main():
    # Initialize components
    confluence_loader = ConfluenceLoader()
    data_processor = DataProcessor()
    vectorstore_manager = VectorStoreManager()
    llm_manager = LLMManager()

    # Step 1: Load Data from Confluence
    print("Loading data from Confluence...")
    raw_documents = confluence_loader.get_all_pages(space_key=os.getenv("SPACE_KEY"))

    # Step 2: Process Data
    print("Processing documents...")
    processed_documents = data_processor.process_documents(raw_documents)

    # Step 3: Create or Load Vectorstore
    print("Creating vectorstore...")
    vectorstore_manager.create_vectorstore(processed_documents)

    # Step 4: Retrieve Relevant Documents
    query = input("Enter your question: ")
    retriever = DocumentRetriever(vectorstore_manager.vectorstore)
    relevant_docs = retriever.retrieve(query)

    # Combine retrieved documents into a context
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Step 5: Generate Answer
    print("Generating answer...")
    answer = llm_manager.generate_answer(context, query)
    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()
