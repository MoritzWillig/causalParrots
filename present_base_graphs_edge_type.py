import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from causalFM.answer_helpers import get_response_flags, categorize_answers, load_compact_answers, adj_mat_to_list
from causalFM.plot import plot_from_adj_mat
from causalFM.query_helpers import load_query_instances, question_templates

save_fig = True
test_run = False  # if true stops after first plot

evaluations_dir = Path("./evaluations")
base_name = "base_edge_type"

base_dir = evaluations_dir / base_name
base_dir.mkdir(exist_ok=True)


from_apis = ["openai", "aleph_alpha", "opt"]
datasets = ["altitude", "causal_health", "driving", "recovery", "cancer", "earthquake"]
dataset_labels = ["Altitude", "Health", "Driving", "Recovery", "Cancer", "Earthquake"]

allow_quiz_answers = True  # include quiz-style answers
positive_response_flags, negative_response_flags, undecided_response_flags = get_response_flags(allow_quiz_answers)

d_adj_mats = []
d_variable_names = []
d_queries = []

for dataset in datasets:
    # adj_mat.shape = [NUM_APIS, NUM_TEMPLATES, FROM_VAR, TO_VAR]
    adj_mats, variable_names, queries = load_compact_answers(dataset, from_apis, len(question_templates), positive_response_flags, negative_response_flags, undecided_response_flags)
    d_adj_mats.append(adj_mats)
    d_variable_names.append(variable_names)
    d_queries.append(queries)

for api_idx, api in enumerate(from_apis):
    for template_idx, template in enumerate(question_templates):
        rows = 1
        cols = len(datasets)

        subplot_size = 5
        fig, axs = plt.subplots(rows, cols, figsize=(subplot_size * cols, subplot_size))
        fig.suptitle(f'API: {api}; Question template: {template["name"]}', y=1)

        for dataset_idx, dataset in enumerate(datasets):
            ax = axs[dataset_idx]
            ax.set_xticks([], [])
            ax.set_yticks([], [])
            ax.set_aspect('equal')
            plt.sca(ax)
            ax.set_title(f'{dataset_labels[dataset_idx]}', y=1)

            adj_mats = d_adj_mats[dataset_idx] #[api_idx, :, :, :]
            variable_names = d_variable_names[dataset_idx]
            queries = d_queries[dataset_idx]

            adj_mat = adj_mats[api_idx, template_idx, :, :]

            def get_edge_type_label(a, i, j):
                et = a[i, j]
                if et > 0:
                    return "+"
                elif et == 0:
                    return "-"
                elif et < 0:
                    return "?"

            adj_labels = adj_mat_to_list(adj_mat, get_edge_type_label)
            plot_from_adj_mat(
                adj_mat, variable_names, dataset, ax=ax, abrev_vars=True, edge_labels=adj_labels,
                edge_mode='sign'
            )

        graph_name = f"{base_name}_{api}_{template_idx}{template['name']}"
        fig.tight_layout(h_pad=0.5)

        if save_fig:
            plt.savefig(base_dir / f"{graph_name}.png", dpi=200)
        else:
            plt.show()

        if test_run:
            exit()
