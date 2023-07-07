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


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text  # or whatever


def remove_suffix(text, suffix):
    if text.endswith(suffix):
        return text[:-len(suffix)]
    return text


def sanitize_QA(query_text):
    # remove "Q: A:" style
    query_text = remove_prefix(query_text, "Q: ")
    query_text = remove_prefix(query_text, "Q:")
    query_text = remove_prefix(query_text, "A:")
    query_text = remove_prefix(query_text, "A: ")
    return query_text


def query_openai_gpt_4(context, query_text: str, dry_run=False) -> Optional[str]:
    if dry_run:
        return None

    query_text = [sanitize_QA(line) for line in query_text.split("\n")]
    query_text = [line for line in query_text if line != ""]

    messages = [
        *[{"role": "user" if i % 2 == 0 else "assistant", "content": line} for i, line in enumerate(query_text)]
    ]
    response = openai.ChatCompletion.create(
        # engine=self.model_name,
        model="gpt-4",
        messages=messages,
        temperature=0,
        max_tokens=50,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]["message"]["content"]


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
            engine="text-embedding-ada-002"  # 12288
        )
        embedding = response["data"][0]["embedding"]
        return embedding

    def do_get_embedding_batch(self, query_text: List[str]) -> list:
        response = openai.Embedding.create(
            input=query_text,
            engine="text-embedding-ada-002"  # 12288
        )

        embeddings = [""]*len(query_text)
        for answer in response["data"]:
            embeddings[answer.index] = answer.embedding
        return embeddings
