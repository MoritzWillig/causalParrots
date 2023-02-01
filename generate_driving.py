import pickle
import re
from itertools import chain, zip_longest
from pathlib import Path
from typing import Union

from causalFM.query_helpers import questions, AttrDict, instantiate_questions, store_query_instances

dry_run = False
queries_path = "./queries/driving"

variables = [
    AttrDict.make({
        "name": "fuel",
        "expression": "remaining fuel",
        "singular": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "driveStyle",
        "expression": "driving style",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "carType",
        "expression": "car type",
        "singular": True,
        "optionalThe": True,
        "alt": []
    })
]


def main():
    question_instances = instantiate_questions(questions, variables)
    if not dry_run:
        store_query_instances(queries_path, question_instances)
    print("done.")


if __name__ == "__main__":
    main()
