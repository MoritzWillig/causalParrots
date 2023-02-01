import pickle
import re
from itertools import chain, zip_longest
from pathlib import Path
from typing import Union

from causalFM.query_helpers import questions, AttrDict, instantiate_questions, store_query_instances

dry_run = False
queries_path = "./queries/cancer"

variables = [
    AttrDict.make({
        "name": "pollution",
        "expression": "pollution",
        "singular": True,
        "alt": [
            {
                "name": "envPollution",
                "expression": "environmental pollution",
                "singular": True
            }
        ]
    }),
    AttrDict.make({
        "name": "smoking",
        "expression": "smoking",
        "singular": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "cancer",
        "expression": "cancer",
        "singular": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "x-ray",
        "expression": "x-ray examinations",
        "singular": False,
        "alt": []
    }),
    AttrDict.make({
        "name": "dyspnoea",
        "expression": "dyspnoea",
        "singular": True,
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
