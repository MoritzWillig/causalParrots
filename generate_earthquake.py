import pickle
import re
from itertools import chain, zip_longest
from pathlib import Path
from typing import Union

from causalFM.query_helpers import questions, AttrDict, instantiate_questions, store_query_instances

dry_run = False
queries_path = "./queries/earthquake"

variables = [
    AttrDict.make({
        "name": "earthquake",
        "expression": "earthquakes",
        "singular": False,
        "alt": []
    }),
    AttrDict.make({
        "name": "burglaries",
        "expression": "burglaries",
        "singular": False,
        "alt": [
            {
                "name": "intruders",
                "expression": "intruders",
                "singular": False
            }
        ]
    }),
    AttrDict.make({
        "name": "alarms",
        "expression": "alarms",
        "singular": False,
        "alt": [
            {
                "name": "burglarAlarm",
                "expression": "burglary alarms",
                "singular": False
            }
        ]
    }),
    AttrDict.make({
        "name": "john",
        "expression": "calls from John",
        "singular": False,
        "alt": []
    }),
    AttrDict.make({
        "name": "mary",
        "expression": "calls from Mary",
        "singular": False,
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
