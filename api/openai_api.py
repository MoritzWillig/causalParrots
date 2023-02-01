from pathlib import Path
from typing import Optional, List

import openai

from causalFM.api.languageModelInterface import LanguageModelInterface


def startup_openai(key):
    if isinstance(key, Path):
        with open(key / "openai", "r") as f:
            openai.api_key = f.readline()  # os.getenv("OPENAI_API_KEY")
    elif isinstance(key, str):
        openai.api_key = key
    else:
        raise ValueError("unknown parameter type.")
    return None


def query_openai(context, query_text, dry_run=False):
    if dry_run:
        return None
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=query_text,
        temperature=0,
        max_tokens=50, #50
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['text']


class OpenAILM(LanguageModelInterface):

    def __init__(self, key: str = None, limit: Optional[int] = 1000, dry_run: bool = False):
        # limit for free account is 20/min, 60/min for paid in the first 48h and 3000 afterwards.
        super().__init__(key, limit, dry_run)
        self.context = startup_openai(key)

    def do_query(self, query_text: str) -> str:
        return query_openai(self.context, query_text)

    def do_get_embedding(self, query_text: str) -> list:
        response = openai.Embedding.create(
            input=query_text,
            engine="text-similarity-davinci-001"  # 12288
        )
        embedding = response["data"][0]["embedding"]
        return embedding


class OpenAILMAda(LanguageModelInterface):

    def __init__(self, key: str = None, limit: Optional[int] = 1000, dry_run: bool = False):
        # limit for free account is 20/min, 60/min for paid in the first 48h and 3000 afterwards.
        super().__init__(key, limit, dry_run)
        self.context = startup_openai(key)

    def do_query(self, query_text: str) -> str:
        #return query_openai(self.context, query_text)
        raise NotImplementedError()

    def do_get_embedding(self, query_text: str) -> list:
        response = openai.Embedding.create(
            input=query_text,
            engine="text-embedding-ada-002"
        )
        embedding = response["data"][0]["embedding"]
        return embedding

    def do_get_embedding_batch(self, query_text: List[str]) -> list:
        response = openai.Embedding.create(
            input=query_text,
            engine="text-embedding-ada-002"
        )

        embeddings = [""]*len(query_text)
        for answer in response["data"]:
            embeddings[answer.index] = answer.embedding
        return embeddings
