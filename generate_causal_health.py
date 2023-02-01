import pickle
import re
from itertools import chain, zip_longest
from pathlib import Path
from typing import Union

from causalFM.query_helpers import questions, AttrDict, instantiate_questions, store_query_instances

dry_run = False
queries_path = "./queries/causal_health"

# if true alternative variable expressions are queried
generate_alternatives = True

variables = [
    AttrDict.make({
        "name": "age",
        "expression": "age",
        "singular": True,
        "alt": [
            {
                "name": "aging",
                "expression": "aging",
                "singular": True
            }
        ]
    }),
    AttrDict.make({
        "name": "health",
        "expression": "health",
        "singular": True,
        "alt": [
            {
                "name": "conditions",
                "expression": "health conditions",
                "singular": False
            },
            {
                "name": "healthiness",
                "expression": "healthiness",
                "singular": True,
            }
        ]
    }),
    AttrDict.make({
        "name": "mobility",
        "expression": "mobility",
        "singular": True,
        "alt": [
            {
                "name": "fitness",
                "expression": "fitness",
                "singular": True
            },
            {
                "name": "agility",
                "expression": "agility",
                "singular": True
            }
        ]
    }),
    AttrDict.make({
        "name": "nutrition",
        "expression": "nutrition",
        "singular": True,
        "alt": [
            {
                "name": "habits",
                "expression": "eating habits",
                "singular": False
            },
            {
                "name": "diet",
                "expression": "a diet",
                "singular": True
            }
        ]
    })
]


def generate_queries(queries_path, variables, constrain_to_var=None):
    question_instances = instantiate_questions(questions, variables, constrain_to_var=constrain_to_var)
    if not dry_run:
        store_query_instances(queries_path, question_instances)


def main():
    if generate_alternatives:
        for alt_variable in variables:
            var_name = alt_variable["name"]
            for alternative in alt_variable["alt"]:
                assert var_name != alternative["name"] # make sure that the alternative name differs

                # build up a new variables configuration
                alt_vars_config = []
                for variable in variables:
                    if variable == alt_variable:
                        alt_var_config = AttrDict.make({
                            "name": var_name,
                            "alt_name": alternative["name"],
                            "expression": alternative["expression"],
                            "singular": alternative["singular"],
                            "optionalThe": alternative.get("optionalThe", False)
                        })
                        alt_vars_config.append(alt_var_config)
                    else:
                        # keep all other var configs untouched
                        alt_vars_config.append(variable)
                # generate & save ...
                alt_queries_path = f"{queries_path}__alt_{alt_variable['name']}_{alternative['name']}"
                generate_queries(alt_queries_path, alt_vars_config, constrain_to_var=var_name)  # only query edges that contain the altered var
    else:
        generate_queries(queries_path, variables)
    print("done.")


if __name__ == "__main__":
    main()
