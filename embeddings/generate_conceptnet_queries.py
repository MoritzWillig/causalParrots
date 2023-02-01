from pathlib import Path
import json

from causalFM.query_helpers import statement_templates, AttrDict, instantiate_questions, store_query_instances, \
    instantiate_question_pairs, question_templates

dry_run = False
queries_base_path = Path("./queries/")

cause_sets_path = Path("./conceptnet_cause_sets.txt")


def concept_to_string(c):
    return c.replace("_", " ")


def read_causal_concepts(path: Path):
    edges = []
    with path.open("r") as f:
        lines = f.readlines()
    for line in lines:
        cause, effects = line.rstrip().split(",", maxsplit=1)
        cause = concept_to_string(cause)
        effects = [concept_to_string(effect) for effect in json.loads(effects)]

        edges.extend([(cause, effect) for effect in effects])
    return edges


def generate_pairs(edges):
    for i, edge in enumerate(edges):
        info = [
            AttrDict.make({
                "name": edge[0],
                "expression": edge[0],
                "singular": True,
            }),
            AttrDict.make({
                "name": edge[1],
                "expression": edge[1],
                "singular": True,
                #"optionalThe": False,
            })
        ]
        edges[i] = info
    return edges


ignore_modifiers = ["QA", "YesNo"]  # None

sentence_template = statement_templates  # question_templates

datasets = {
    "causal_concepts": {"type": "pairs", "instances": generate_pairs(read_causal_concepts(cause_sets_path))},
}


def generate_queries(queries_path, dataset):
    """
    :param queries_path:
    :param variables:
    :param constrain_to_var: if querying alternative var names - only query edges that are affected by the altered var name
    :return:
    """
    question_instances = instantiate_question_pairs(
        sentence_template, dataset["instances"],
        constrain_to_var=None,
        prevent_modfiers=ignore_modifiers,
        generate_inverse=dataset.get("generate_inverse", True))
    if not dry_run:
        store_query_instances(queries_path, question_instances)


def main():
    for dataset_name, dataset in datasets.items():
        print(f"generating dataset: {dataset_name}")
        generate_queries(
            queries_base_path / dataset_name,
            dataset)

    print("done.")


if __name__ == "__main__":
    main()
