import os
import pickle
from pathlib import Path
from typing import Union
import re
import numpy as np


def get_response_flags(allow_quiz_answers):
    # positive/negative/undecided
    all_response_flags = ["y", "n", "yp", "np", "yi", "ni", "yo", "no", "ye", "ne", "u", "uq", "uqy", "uqn", "wd", "x", "meta"]
    positive_response_flags = ["y", "yp", "yi", "yo", "ye"] + ["uqy"] if allow_quiz_answers else []
    negative_response_flags = ["n", "np", "ni", "no", "ne"] + ["uqn"] if allow_quiz_answers else []
    undecided_response_flags = ["u", "uq", "wd", "x"] + [] if allow_quiz_answers else ["uqy", "uqn"]
    meta_response_flags = ["meta"]
    assert len(positive_response_flags) == len(negative_response_flags)
    assert len(positive_response_flags) + len(negative_response_flags) + len(
        undecided_response_flags) + len(meta_response_flags) == len(all_response_flags)
    return positive_response_flags, negative_response_flags, undecided_response_flags, meta_response_flags


def categorize_answers(answers):
    positive_response_flags, negative_response_flags, undecided_response_flags, meta_response_flags = get_response_flags(allow_quiz_answers=True)

    ca = []
    for answer in answers:
        if answer in positive_response_flags:
            ca.append(1)
        elif answer in negative_response_flags:
            ca.append(0)
        elif answer in undecided_response_flags:
            ca.append(-1)
        elif answer in meta_response_flags:
            #ca.append(float('inf'))# 1j
            ca.append(1j)
        else:
            raise RuntimeError(f"Unknown response flag: {answer}")
    return ca


def _load_answers(dataset_name, apis, num_query_templates):
    # queries = load_query_instances(Path(f"./queries/{dataset_name}_questions.txt"))
    with Path(f"./queries/{dataset_name}_full.pkl").open("rb") as f:
        queries = pickle.load(f)

    # query format:
    # question: str
    # "info": template:str, names:[A,B], alt_names:[None|str, None|str], exprs:[str, str]

    api_responses = []
    for api_name in apis:
        answer_dir = Path(f"./queries/{api_name}_{dataset_name}")
        summary_path = answer_dir / "summary.txt"
        with summary_path.open("r") as f:
            answers = f.read().splitlines()

        canswers = categorize_answers(answers)
        # note: assumes same pair ordering for all templates
        canswers = np.array(canswers).reshape((num_query_templates, -1))  # reshape into [#TEMPLATES, var_combs]
        api_responses.append(canswers)

    # [API, NUM_TEMPLATES, VAR_COMBS]
    full_responses = np.array(api_responses, dtype=np.complex_)

    variable_names = []
    for query in queries:
        names = query["info"]["names"]
        if names[0] not in variable_names:
            variable_names.append(names[0])
        if names[1] not in variable_names:
            variable_names.append(names[1])

    return full_responses, queries, variable_names


def load_compact_answers(dataset_name, apis, num_query_templates):
    full_responses, queries, variable_names = _load_answers(
        dataset_name, apis, num_query_templates)
    var_to_idx = {name: i for i, name in enumerate(variable_names)}

    num_variables = len(variable_names)
    adj_mat = np.zeros((len(apis), num_query_templates, num_variables, num_variables), dtype=np.complex_)
    # note: assumes same pair ordering for all templates
    for i in range((num_variables-1)*num_variables):
        query = queries[i]
        names = query["info"]["names"]
        v0_idx = var_to_idx[names[0]]
        v1_idx = var_to_idx[names[1]]
        adj_mat[:, :, v0_idx, v1_idx] = full_responses[:, :, i]

    #adj_mat.shape = [NUM_APIS, NUM_TEMPLATES, FROM_VAR, TO_VAR]
    return adj_mat, variable_names, queries


def load_alternative(alternative_dataset_name, adj_mat, variable_names, apis, num_query_templates):
    adj_mat = np.copy(adj_mat)
    alt_full_responses, alt_queries, alt_variable_names = _load_answers(
        alternative_dataset_name, apis, num_query_templates)
    var_to_idx = {name: i for i, name in enumerate(variable_names)}

    alt_names = set()
    altered_names = set()

    num_variables = len(alt_variable_names)
    mask = np.zeros((num_variables, num_variables), dtype=np.bool)
    # note: assumes same pair ordering for all templates
    # we query the alternative with each other var (n-1), in both directions (*2)
    for i in range((num_variables-1)*2):
        query = alt_queries[i]
        names = query["info"]["names"]
        v0_idx = var_to_idx[names[0]]
        v1_idx = var_to_idx[names[1]]
        adj_mat[:, :, v0_idx, v1_idx] = alt_full_responses[:, :, i]
        mask[v0_idx, v1_idx] = True

        # FIXME it would be nicer if we would store (name,altname), but at the moment we
        # assume that at most one alt_name is present
        q_alt_names = query["info"]["alt_name"]
        if q_alt_names[0] is not None:
            alt_names.add(q_alt_names[0])
            altered_names.add(names[0])
        if q_alt_names[1] is not None:
            alt_names.add(q_alt_names[1])
            altered_names.add(names[1])

    return adj_mat, mask, list(alt_names), list(altered_names)


def adj_mat_to_list(adj_mat, func):
    row = []
    for i in range(adj_mat.shape[0]):
        cols = []
        for j in range(adj_mat.shape[1]):
            cols.append(func(adj_mat, i, j))
        row.append(cols)

    return row
