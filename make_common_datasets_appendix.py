from pathlib import Path
import numpy as np
import json

from causalFM.generate_causal_chains import causal_chains_data, causal_chains_questions
from causalFM.generate_intuitive_physics import physics_data, physics_questions
from causalFM.query_helpers import load_query_instances
import re

evaluations_dir = Path("./evaluations")
base_name = "common"

base_dir = evaluations_dir / base_name
base_dir.mkdir(exist_ok=True)

from_apis = ["openai", "aleph_alpha", "opt"]
api_labels = {
    "openai": "GPT-3",
    "aleph_alpha": "Luminous",
    "opt": "OPT"
}

datasets = ["intuitive_physics", "causal_chains"]
dataset_labels = ["Intuitive Physics", "Propositional Logic"]

process = {
    "chain": "order_len"
}

group_names = {
    "chain": "Causal Chains",
    "sub": "Sub chains",
    "rand": "Randomized Variable",
    "rolling": "Rolling",
    "support": "Support",
    "collisions": "Collisions",
    "weights": "Weights",
    "mechanisms": "Mechanisms"
}

dataset_datas={
    "causal_chains": [causal_chains_data, causal_chains_questions],
    "intuitive_physics": [physics_data, physics_questions]
}


def sanitize_answer(a: str):
    end_text = None

    a = a.replace('\r', '')
    a = a.replace('\t', ' ')
    a = a.replace('$', '\\$')
    a = a.replace('&', '\&')
    a = a.replace('\u2212', '-')
    a = re.sub(r'\n+', '\n ', a)
    lines = []
    for l in a.split("\n"):
        l = l.strip()
        if l.startswith("."):
            l = l[2:]
        l = l.strip()
        if l != "":
            lines.append(l)
    if len(lines) == 0:
        end_text = "\\textit{[empty]}"
    else:
        last = lines[-1]
        off = 0
        for i in range(len(lines)-1):
            line = lines[-1-i+off]
            ml = min(len(line), len(last))
            if ml > 3 and line[:ml] == last[:ml]:
                last = line # last lines are often cut off, so we use earlier lines
                lines.pop()
                off += 1
                continue
            break
        if off != 0:
            lines.append(last)  # we deleted all repetitions, reappend text
            end_text = "\\textit{[repeating]}"

    max_lines = 2
    if len(lines) > max_lines:
        # even if there is another end text we cut and overwrite if we get more lines remaining
        lines = lines[:max_lines]
        end_text = "\\textit{[continued]}"

    if end_text is not None:
        lines.append(end_text)

    s = " \\newline ".join(lines)
    return s


for dataset_name, dataset_label in zip(datasets, dataset_labels):

    groups = {}

    dataset_data, dataset_questions = dataset_datas[dataset_name]
    print("\\clearpage")
    print("\\section{"+dataset_label+"}")

    # load and sort data
    with (base_dir / f"{dataset_name}_compact.json").open("r") as f:
        data = json.load(f)

        queries = data["queries"]

        for query in queries:
            idx = dataset_questions.index(query["question"])
            group = dataset_data[idx][0]
            if not group in groups:
                groups[group] = []
            groups[group].append([query["question"], query["responses"]])

        for name, group in groups.items():
            if name in process:
                if process[name] == "order_len":
                    groups[name] = sorted(group, key=lambda entry: len(entry[0]))

    # print groups
    for group in groups:
        print("\\subsection{" + dataset_label + " - " + group_names[group] + "}")

        for i, query in enumerate(queries):
            question = query["question"]
            responses = query["responses"]

            text = "\\begin{tabular}{|p{1.5cm} p{13cm}|}\n"
            text += "\\hline\n"
            text += f"  \\multicolumn{{2}}{{|p{{14.5cm}}|}}{{``{question}''}}\\\\\n"
            text += "\\hline\n"
            for api, response in responses.items():
                text += f"  {api_labels[api]} & {sanitize_answer(response)}\\\\\n"
            text += "\\hline\n"
            text += "\\end{tabular}\n"

            print(text)
