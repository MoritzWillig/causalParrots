from pathlib import Path

from causalFM.query_helpers import statement_templates, AttrDict, instantiate_questions, store_query_instances, \
    instantiate_question_pairs, question_templates

dry_run = False
queries_base_path = Path("./queries/")
generate_alternatives = True

altitude_variables = [
    AttrDict.make({
        "name": "altitude",
        "expression": "altitude",
        "singular": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "temperature",
        "expression": "temperature",
        "singular": True,
        #"optionalThe": True,
        "alt": []
    })
]


cancer_variables = [
    AttrDict.make({
        "name": "pollution",
        "expression": "pollution",
        "singular": True,
        "alt": []
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


causal_health_variables = [
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


driving_variables = [
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


earthquake_variables = [
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
        "alt": []
    }),
    AttrDict.make({
        "name": "alarms",
        "expression": "alarms",
        "singular": False,
        "alt": []
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


recovery_variables = [
    AttrDict.make({
        "name": "treatment",
        "expression": "treatments",
        "singular": False,
        "alt": []
    }),
    AttrDict.make({
        "name": "recovery",
        "expression": "speed of recovery",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "precondition",
        "expression": "preconditions",
        "singular": False,
        "optionalThe": False,
        "alt": []
    })
]

answer_statements = [
    "Yes.",
    "No.",
    "There is a causal relation.",
    "There is no causal relation.",
    "This is true.",
    "This is false.",
    "This is correct.",
    "This is incorrect."
]

causal_statements = [
    [ #"A monument causes visitors.", "Visitors cause a monument.",
        AttrDict.make({
            "name": "monument",
            "expression": "a monument",
            "singular": True,
        }),
        AttrDict.make({
            "name": "visitors",
            "expression": "visitors",
            "singular": False,
            #"optionalThe": False,
        })
    ],
    [
        AttrDict.make({
            "name": "sunnyDay",
            "expression": "A sunny day",
            "singular": True,
        }),
        AttrDict.make({
            "name": "happiness",
            "expression": "happiness",
            "singular": True,
        })
    ],
    [
        AttrDict.make({
            "name": "heat",
            "expression": "heat",
            "singular": True,
        }),
        AttrDict.make({
            "name": "melting",
            "expression": "melting ice",
            "singular": True,
        })
    ],
    [
        AttrDict.make({
            "name": "material",
            "expression": "source material",
            "singular": True,
            "optionalThe": True
        }),
        AttrDict.make({
            "name": "weight",
            "expression": "the object‘s weight",
            "singular": True,
            #"optionalThe": True
        })
    ],
    [
        AttrDict.make({
            "name": "doctor",
            "expression": "a doctor's visit",
            "singular": True,
        }),
        AttrDict.make({
            "name": "diagnoses",
            "expression": "diagnoses",
            "singular": False,
        })
    ],
    [
        AttrDict.make({
            "name": "cold",
            "expression": "a cold",
            "singular": True,
        }),
        AttrDict.make({
            "name": "coughing",
            "expression": "coughing",
            "singular": True,
        })
    ],
    [
        AttrDict.make({
            "name": "traffic",
            "expression": "traffic",
            "singular": True,
        }),
        AttrDict.make({
            "name": "pollution",
            "expression": "air pollution",
            "singular": True,
        })
    ],
    [
        AttrDict.make({
            "name": "holidays",
            "expression": "holidays",
            "singular": False,
        }),
        AttrDict.make({
            "name": "flights",
            "expression": "flights",
            "singular": False,
        })
    ],
    [
        AttrDict.make({
            "name": "prices",
            "expression": "increasing prices",
            "singular": False,
        }),
        AttrDict.make({
            "name": "demand",
            "expression": "lowered demand",
            "singular": True,
        })
    ],
    [
        AttrDict.make({
            "name": "economy",
            "expression": "A strong economy",
            "singular": True,
        }),
        AttrDict.make({
            "name": "wealth",
            "expression": "wealth of a country",
            "singular": True,
            "optionalThe": True
        })
    ],
]


causal_chains = [
    "A causes B and B causes C. A causes C.",  # A->B->C. A->C.
    "A causes B and B causes C. A causes B.",  # A->B->C. A->B.
    "A causes B and B causes C. B causes C.",  # A->B->C. B->C.
    "A causes B and B causes C. A causes A.",  # A->B->C. A->A.
    "A causes B and B causes C. B causes A.",  # A->B->C. B->A.
    "A causes B and B causes C. C causes A.",  # A->B->C. C->A.

    # extending chain
    "A causes B, B causes C and C causes D. A causes D.",
    "A causes B, B causes C, C causes D and D causes E. A causes E.",
    "A causes B, B causes C, C causes D, D causes E, E causes F. A causes F.",
    "A causes B, B causes C, C causes D, D causes E, E causes F. B causes E.",
    "A causes B, B causes C, C causes D, D causes E, E causes F. E causes B.",

    # changing clause order
    "B causes C and A causes B. A causes C.",  # B->C, A->B. A->C.
    "B causes C and A causes B. C causes A.",  # B->C, A->B. C->A.

    # changing variable names
    "G causes Q and Q causes S. G causes S.",  # G->Q->S. G->S.

    # changing clause order and rename
    "Q causes S and G causes Q. G causes S."  # Q->S, G->Q. G->S.
]


dataset_lists = {
    "causal_chains": {"type": "list", "instances": causal_chains},
    "answer_statements": {"type": "list", "instances": answer_statements}
}
datasets_templated = {
    "altitude": altitude_variables,
    "cancer": cancer_variables,
    "causal_health": causal_health_variables,
    "driving": driving_variables,
    "earthquake": earthquake_variables,
    "recovery": recovery_variables,
    "causal_statements": {"type": "pairs", "instances": causal_statements},
}

ignore_modifiers = ["QA", "YesNo"]  # None

#sentence_template = question_templates  # question_templates
sentence_template = [question_templates[2]]  # question_templates
prefix = "question_" # questions with "question_" prefix; statement_templates = <empty>

#sentence_template = statement_templates
#sentence_template = [statement_templates[2]]
#prefix = "statement_"

datasets = {
    **datasets_templated
    #**dataset_lists
}

datasets = {f"{prefix}{k}": v for k, v in datasets.items()}



def generate_queries(queries_path, dataset,
                     generate_alternatives=False, _constrain_to_var=None):
    """
    :param queries_path:
    :param variables:
    :param constrain_to_var: if querying alternative var names - only query edges that are affected by the altered var name
    :return:
    """
    if isinstance(dataset, list):
        if generate_alternatives:
            for alt_variable in dataset:
                var_name = alt_variable["name"]
                alt_var = alt_variable["alt"] if "alt" in alt_variable else []
                for alternative in alt_var:
                    assert var_name != alternative["name"]  # make sure that the alternative name differs

                    # build up a new variables configuration
                    alt_vars_config = []
                    for variable in dataset:
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
                    generate_queries(alt_queries_path, alt_vars_config, generate_alternatives=False, _constrain_to_var=var_name)  # only query edges that contain the altered var

        question_instances = instantiate_questions(
            sentence_template, dataset,
            constrain_to_var=_constrain_to_var,
            prevent_modfiers=ignore_modifiers)
    else:
        if isinstance(dataset, dict):
            if dataset["type"] == "list":
                question_instances = [{
                    "question": instance,
                    "info": None
                } for instance in dataset["instances"]]
            elif dataset["type"] == "pairs":
                question_instances = instantiate_question_pairs(
                    sentence_template, dataset["instances"],
                    constrain_to_var=_constrain_to_var,
                    prevent_modfiers=ignore_modifiers,
                    generate_inverse=dataset.get("generate_inverse", True))
            else:
                raise ValueError("unknown dataset type")
        else:
            raise RuntimeError("unknown dataset type.")
    if not dry_run:
        store_query_instances(queries_path, question_instances)


def main():
    for dataset_name, dataset in datasets.items():
        print(f"generating dataset: {dataset_name}")
        generate_queries(
            queries_base_path / dataset_name,
            dataset, generate_alternatives=generate_alternatives)

    print("done.")


if __name__ == "__main__":
    main()
