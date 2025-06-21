from typing import List
import requests
from langchain_core.embeddings import Embeddings


class ModalEmbeddings(Embeddings):
    def __init__(self, url: str, model_name: str, api_key: str = None):
        self.url = url
        self.model_name = model_name
        self.api_key = api_key

    def embed(self, text: List[str]) -> List[List[str]]:
        # We remove newlines from the text to avoid issues with the embedding model.
        cleaned_text = [t.replace("\n", " ") for t in text]

        payload = {'text': "\n".join(cleaned_text)}

        headers = {}
        if self.api_key:
            headers = {'x-api-key': self.api_key}

        response = requests.post(
            self.url,
            data=payload,
            files=[],
            headers=headers
        )
        response.raise_for_status()

        # print(response.text)
        return response.json()

    def embed_documents(self, text: List[str]) -> List[List[str]]:
        """
        Embed a list of documents using the embedding model.
        """
        return self.embed(text)

    def embed_query(self, text: str) -> List[str]:
        """
        Embed a query
        """
        return self.embed([text])[0]

    def get_model_name(self) -> str:
        return self.model_name


if __name__ == "__main__":
    embeds = ModalEmbeddings(
        url="https://lfoppiano--intfloat-multilingual-e5-large-instruct-embed-5da184.modal.run/",
        model_name="intfloat/multilingual-e5-large-instruct"
    )

    print(embeds.embed(
        ["We are surrounded by stupid kids",
         "We are interested in the future of AI"]
    ))
