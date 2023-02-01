import pickle
import random
from pathlib import Path
import json

from causalFM.query_helpers import statement_templates, AttrDict, instantiate_questions, store_query_instances, \
    instantiate_question_pairs, question_templates

dry_run = False
queries_base_path = Path("./queries/")

seed = 12345

dataset = "causal_concepts"
sample_indices = 10000

cause_sets_path = queries_base_path / f"{dataset}_full.pkl"
cause_sets_idcs_path = queries_base_path / f"{dataset}_{sample_indices}.idcs"


def main():
    with cause_sets_path.open("rb") as f:
        data = pickle.load(f)

    num_grouped = 2
    num_templates = len(statement_templates)
    reduction_factor = num_grouped * num_templates  # number of indices that are generated from a single concept

    num_indices = len(data)
    num_concepts = num_indices // reduction_factor

    random.seed(seed)
    concept_samples = random.sample(range(num_concepts), sample_indices // reduction_factor)

    indices = [""]*(len(concept_samples)*reduction_factor)
    j = 0
    for i, concept_idx in enumerate(concept_samples):
        for t in range(num_templates):
            for g in range(num_grouped):
                indices[j] = str(t*num_grouped*num_concepts + concept_idx*num_grouped + g)
                j += 1

    content = ",\n".join(indices)
    with cause_sets_idcs_path.open("w+") as f:
        f.write(content)




if __name__ == "__main__":
    main()
