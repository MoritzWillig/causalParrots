from pathlib import Path
import numpy as np
import json

from causalFM.query_helpers import load_query_instances

evaluations_dir = Path("./evaluations")
base_name = "common"

base_dir = evaluations_dir / base_name
base_dir.mkdir(exist_ok=True)

from_apis = ["gpt_4", "openai", "aleph_alpha", "opt"]
#from_apis = ["openai", "aleph_alpha"]
#datasets = ["intuitive_physics", "causal_chains"]
#datasets = [f"causal_chains_cot_{i}" for i in range(1,9)]
datasets = ["causal_world", *[f"causal_world_cot_{i}" for i in range(1, 5)]]


for dataset_name in datasets:
    print(f"[{dataset_name}]")

    queries_data = []
    data = {
        "dataset": dataset_name,
        "queries": queries_data
    }

    queries = load_query_instances(Path(f"./queries/{dataset_name}_questions.txt"))
    for i, query in enumerate(queries):

        responses = {}
        qdata = {
            "question": query,
            "responses": responses
        }
        for api_name in from_apis:
            responses[api_name] = Path(f"./queries/{api_name}_{dataset_name}/{i}.txt").read_text()

        queries_data.append(qdata)

    with (base_dir / f"{dataset_name}_compact.json").open("w+") as f:
        f.write(json.dumps(data, indent=4))
