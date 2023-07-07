import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from causalFM.answer_helpers import load_compact_answers, adj_mat_to_list, \
    load_alternative
from causalFM.plot import plot_from_adj_mat
from causalFM.query_helpers import question_templates

save_fig = True
test_run = False  # if true stops after first plot

evaluations_dir = Path("./evaluations")
base_name = "wording_stability"

base_dir = evaluations_dir / base_name
base_dir.mkdir(exist_ok=True)


#from_apis = ["openai", "aleph_alpha", "opt", "gpt_4"]
from_apis = ["gpt_4"]
dataset = "causal_health"
dataset_labels = ["Health"]

alternatives = [
    "causal_health__alt_age_aging",
    "causal_health__alt_health_conditions",
    "causal_health__alt_health_healthiness",
    "causal_health__alt_mobility_agility",
    "causal_health__alt_mobility_fitness",
    "causal_health__alt_nutrition_diet",
    "causal_health__alt_nutrition_habits"
]

allow_quiz_answers = True  # include quiz-style answers

alt_adj_mats = []
alt_masks = []
alt_names = []
altered_names = []

alt_names_set = set(alt_names)

adj_mats, variable_names, queries = load_compact_answers(dataset, from_apis, len(question_templates))

for alternative in alternatives:
    # adj_mat.shape = [NUM_APIS, NUM_TEMPLATES, FROM_VAR, TO_VAR]
    alt_adj_mat, alt_mask, alt_name, altered_name = load_alternative(alternative, adj_mats, variable_names, from_apis, len(question_templates))
    alt_adj_mats.append(alt_adj_mat)
    alt_masks.append(alt_mask)
    alt_names.append(alt_name[0])
    altered_names.append(altered_name[0])

for api_idx, api in enumerate(from_apis):

    rows = 1

    # assuming all variables where altered at some point
    for altered_var in variable_names:

        # find the indices where the current variable was altered
        indices = []
        for i in range(len(altered_names)):
            if altered_names[i] == altered_var:
                indices.append(i)

        cols = len(indices)

        subplot_size = 5
        fig, axs = plt.subplots(rows, cols, figsize=(subplot_size * cols, subplot_size))
        fig.suptitle(f'API: {api}; Stability of "{altered_var}" variable wording', y=1)

        for idc, i in enumerate(indices):
            alternative_dataset = alternatives[i]

            alt_adj_mat = alt_adj_mats[i][api_idx, :, :, :]
            alt_mask = alt_masks[i]
            alt_name = alt_names[i]
            altered_name = altered_names[i]

            ax = axs[idc] if isinstance(axs, np.ndarray) else axs
            ax.set_xticks([], [])
            ax.set_yticks([], [])
            ax.set_aspect('equal')
            plt.sca(ax)
            ax.set_title(f'{alt_name}', y=1)

            adj_mat = adj_mats[api_idx, :, :, :]
            adj_mat = adj_mat.clip(min=0.0)  # set 'unknown' edges to NO
            adj_mat = np.sum(adj_mat, axis=0)  # count number of times the edges appear
            alt_adj_mat = alt_adj_mat.clip(min=0.0)  # set 'unknown' edges to NO
            alt_adj_mat = np.sum(alt_adj_mat, axis=0)  # count number of times the edges appear

            diff = alt_adj_mat - adj_mat  # compute difference between base and altered graph

            diff[~alt_mask] = np.NAN  # ignore unaffected edges

            def signed_nr_str(x):
                if x == 0:
                    return "0"
                elif x > 0:
                    return f"+{x}"
                elif np.isnan(x):
                    return "NaN"
                else:
                    return f"{x}"

            adj_labels = adj_mat_to_list(diff, lambda a, i, j: f"{signed_nr_str(a[i,j])}")
            diff /= (len(question_templates) / 2)
            diff = diff.clip(-1, 1)

            plot_from_adj_mat(
                diff, variable_names, dataset,
                ax=ax, abrev_vars=True,
                edge_labels=adj_labels, edge_mode="diverging")

        graph_name = f"{base_name}_{api}_{altered_var}"
        fig.tight_layout(h_pad=0.5)

        if save_fig:
            plt.savefig(base_dir / f"{graph_name}.png", dpi=200)
        else:
            plt.show()

        if test_run:
            exit()
