import os
from typing import List

from langchain_community.document_loaders import (
    ConfluenceLoader as LangchainConfluenceLoader,
)
from langchain_core.documents import Document


class ConfluenceLoader:
    def __init__(self):
        url = os.getenv("CONFLUENCE_CLOUD_URL")
        username = os.getenv("CONFLUENCE_CLOUD_USER")
        api_key = os.getenv("CONFLUENCE_CLOUD_TOKEN")

        if not url:
            raise ValueError("CONFLUENCE_CLOUD_URL environment variable is not set")
        if not username:
            raise ValueError("CONFLUENCE_CLOUD_USER environment variable is not set")
        if not api_key:
            raise ValueError("CONFLUENCE_CLOUD_TOKEN environment variable is not set")

        self.loader = LangchainConfluenceLoader(url=url, username=username, api_key=api_key, cloud=True)

    def get_page_content(self, page_id: str) -> Document:
        page = self.loader.confluence.get_page_by_id(page_id, expand="body.storage")
        content = page["body"]["storage"]["value"]
        return Document(page_content=content, metadata={"page_id": page_id, "title": page["title"]})

    def get_page_comments(self, page_id: str) -> List[Document]:
        comments = self.loader.confluence.get_page_comments(page_id, expand="body.view.value", depth="all")["results"]
        comment_documents = []
        for comment in comments:
            body = comment["body"]["view"]["value"]
            created = comment.get("created", "No timestamp")
            comment_documents.append(
                Document(page_content=body, metadata={"page_id": page_id, "created": created, "type": "comment"})
            )
        return comment_documents

    def get_all_pages(self, space_key: str) -> List[Document]:
        pages = self.loader.confluence.get_all_pages_from_space(
            space=space_key, start=0, limit=50, expand="body.storage"
        )
        documents = []
        for page in pages:
            page_id = page["id"]
            title = page["title"]
            content = page["body"]["storage"]["value"]
            documents.append(
                Document(page_content=content, metadata={"page_id": page_id, "title": title, "space_key": space_key})
            )
        return documents

    def get_all_child_pages(self, page_id: str, max_depth: int) -> List[Document]:
        def _get_child_pages(page_id: str, current_depth: int) -> List[Document]:
            if current_depth >= max_depth:
                return []
            child_documents = []
            direct_children = self.loader.confluence.get_page_child_by_type(
                page_id=page_id, type="page", start=None, limit=None, expand="body.storage"
            )
            for child in direct_children:
                child_id = child["id"]
                child_title = child["title"]
                child_content = child["body"]["storage"]["value"]
                child_documents.append(
                    Document(
                        page_content=child_content,
                        metadata={"page_id": child_id, "title": child_title, "depth": current_depth},
                    )
                )
                # Recursively get child pages of the current child
                child_documents.extend(_get_child_pages(child_id, current_depth + 1))
            return child_documents

        return _get_child_pages(page_id, 0)
