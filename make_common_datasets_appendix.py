import os
from pathlib import Path
import numpy as np
import json
import subprocess

from causalFM.generate_causal_chains import causal_chains_data, causal_chains_questions, create_cot_chain_prefix, cot_chain_postfix
from causalFM.generate_causal_world import create_cot_world_prefix, cot_world_postfix, causal_world_data, \
    causal_world_questions
from causalFM.generate_intuitive_physics import physics_data, physics_questions
from causalFM.query_helpers import load_query_instances
import re

evaluations_dir = Path("./evaluations")
base_name = "common"

tex_docs_dir = Path("./media/tex/")
tex_temp_dir_name = "tmp"
tex_temp_dir = tex_docs_dir / tex_temp_dir_name
tex_temp_dir.mkdir(exist_ok=True)

base_dir = evaluations_dir / base_name
base_dir.mkdir(exist_ok=True)

#from_apis = ["openai", "aleph_alpha"]
from_apis = ["gpt-4", "openai", "aleph_alpha", "opt"]
#from_apis = ["gpt-4"]
api_labels = {
    "gpt_4": "GPT-4",
    "openai": "GPT-3",
    "aleph_alpha": "Luminous",
    "opt": "OPT"
}

datasets = [
    "intuitive_physics", "causal_chains",
    *[f"causal_chains_cot_{i}" for i in range(1, 9)],
    "causal_world",
    *[f"causal_world_cot_{i}" for i in range(1, 5)]
]
dataset_labels = [
    "Intuitive Physics", "Propositional Logic",
    *[f"Propositional Logic (Chain of Thought; Prefix Samples {i})" for i in range(1, 9)],
    "Natural Language Concepts",
    *[f"Natural Language Concepts (Chain of Thought; Prefix Samples {i})" for i in range(1, 5)],
]
dataset_comments = [
    "List of questions and answers for intuitive physics.",
    "List of questions and answers for propositional logic.",
    *["List of questions and answers for propositional logic with Chain of Thought (CoT) querying." for i in range(1, 9)],
    "List of questions and answers for causal world queries.",
    *["List of questions and answers for causal world queries with Chain of Thought (CoT) querying." for i in range(1, 5)],
]

prefixes = {
    **{f"causal_chains_cot_{i}": create_cot_chain_prefix(i) for i in range(1, 9)},
    **{f"causal_world_cot_{i}": create_cot_world_prefix(i) for i in range(1, 5)},
}
postfixes = {
    **{f"causal_chains_cot_{i}": cot_chain_postfix for i in range(1, 9)},
    **{f"causal_world_cot_{i}": cot_world_postfix for i in range(1, 5)},
}

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
    'seesaw': "Seesaw",
    "weights": "Weights",
    "tools": "Tools",

    "general": "Real World",
    "im_rotation": "Imaginary Concepts",
    "im_semi": "Mixed Concepts"
}

dataset_datas = {
    "intuitive_physics": [physics_data, physics_questions],
    "causal_chains": [causal_chains_data, causal_chains_questions],
    **{f"causal_chains_cot_{i}": [causal_chains_data, causal_chains_questions] for i in range(1, 9)},
    "causal_world": [causal_world_data, causal_world_questions],
    **{f"causal_world_cot_{i}": [causal_world_data, causal_world_questions] for i in range(1, 5)},
}


tex_head = r"""\documentclass[12pt]{article}

\usepackage[american]{babel}
\usepackage{booktabs} % commands to create good-looking tables
\usepackage{multirow}

\usepackage{makecell}
\usepackage{hyperref}
\usepackage[shortlabels]{enumitem}
\usepackage{framed}
\usepackage[table]{xcolor}

\begin{document}

"""
tex_tail = r"""

\end{document}"""


def sanitize_answer(a: str):
    end_text = None

    a = a.replace('\r', '')
    a = a.replace('\t', ' ')
    a = a.replace('$', '\\$')
    a = a.replace('&', '\&')
    a = a.replace('\u2212', '-')
    a = re.sub(r'\n+', '\n ', a)
    a = a[:450]  # limit answer length
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


ds_total_queries = {}
for dataset_name, dataset_label, dataset_comment in zip(datasets, dataset_labels, dataset_comments):

    prefix = prefixes.get(dataset_name, None)
    if prefix is not None:
        prefix = len(prefix)
    postfix = postfixes.get(dataset_name, None)
    if postfix is not None:
        postfix = -len(postfix.rstrip())


    groups = {}

    dataset_data, dataset_questions = dataset_datas[dataset_name]

    full_text = ""
    full_text += "\\section{"+dataset_label+"}\n"
    full_text += f"{dataset_comment}\n\n"

    # load and sort data
    with (base_dir / f"{dataset_name}_compact.json").open("r") as f:
        data = json.load(f)

        queries = data["queries"]

        total_queries = 0
        for query in queries:
            question = query["question"][prefix:postfix]
            query["trimmed_question"] = question
            idx = dataset_questions.index(question)
            group = dataset_data[idx][0]
            if not group in groups:
                groups[group] = []
            groups[group].append(query)
            total_queries += 1

        for name, group in groups.items():
            if name in process:
                if process[name] == "order_len":
                    groups[name] = sorted(group, key=lambda entry: len(entry["question"]))
    ds_total_queries[dataset_name] = total_queries

    # print groups
    for group_name, group in groups.items():
        full_text += "\\subsection{" + dataset_label + " - " + group_names[group_name] + "}\n"

        for i, query in enumerate(group):
            question = query["trimmed_question"]
            responses = query["responses"]

            text = "\\begin{tabular}{|p{1.5cm} p{13cm}|}\n"
            text += "\\hline\n"
            text += f"  \\multicolumn{{2}}{{|p{{14.5cm}}|}}{{``{question}''}}\\\\\n"
            text += "\\hline\n"
            for api, response in responses.items():
                text += f"  {api_labels[api]} & {sanitize_answer(response)}\\\\\n"
            text += "\\hline\n"
            text += "\\end{tabular}\n"

            full_text += text + "\n"

    full_text = tex_head + full_text + tex_tail

    tex_file_name = f"{dataset_name}.tex"
    with (tex_docs_dir / tex_file_name).open("w+") as f:
        f.write(full_text)

    p = subprocess.Popen(["pdflatex", f"-output-directory={tex_temp_dir_name}", f"{tex_file_name}"], cwd=tex_docs_dir)
    if p.wait() != 0:
        print("\nERROR WITH pdflatex")
        exit()
    p = subprocess.Popen(["mv", f"{tex_temp_dir_name}/{dataset_name}.pdf", f"{dataset_name}.pdf"], cwd=tex_docs_dir)
    if p.wait() != 0:
        print("\nERROR WITH mv")
        exit()


for name, val in ds_total_queries.items():
    print("total queries", name, val)
