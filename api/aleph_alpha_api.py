from pathlib import Path
from aleph_alpha_client import AlephAlphaClient

from causalFM.api.languageModelInterface import LanguageModelInterface


def startup_aleph_alpha(key):
    if isinstance(key, Path):
        with open(key / "aleph_alpha", "r") as f:
            token = f.readline()
    elif isinstance(key, str):
        token = key
    else:
        raise ValueError("unknown parameter type.")
    client = AlephAlphaClient(
        host="https://api.aleph-alpha.com",
        token=token
    )

    return client


def query_aleph_alpha(context, query_text, dry_run=False):
    print("[querying]", query_text)
    if dry_run:
        return None

    model = "luminous-base"

    client = context
    result = client.complete(
        model,
        query_text,
        maximum_tokens=50,
        temperature=0.0,
        top_k=0,
        top_p=0,
        presence_penalty=0,
        frequency_penalty=0
    )

    return result['completions'][0]['completion']


class AlphaAlphaLM(LanguageModelInterface):

    def __init__(self, key: str = None, dry_run: bool = False):
        super().__init__(key, dry_run)
        self.context = startup_aleph_alpha(key)

    def do_query(self, query_text: str) -> str:
        return query_aleph_alpha(self.context, query_text)

    #def do_get_embedding(self, query_text: str) -> list:
    #    raise NotImplementedError()
