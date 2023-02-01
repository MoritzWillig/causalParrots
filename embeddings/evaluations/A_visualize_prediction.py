import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from causalFM.embeddings.evaluations.helpers.embedding_processing import load_embeddings, query_statement_filters, \
    single_template_statement_filters
from causalFM.plot import plot_from_adj_mat
from causalFM.query_helpers import question_templates, statement_templates
from causalFM.statistics_table import make_statistics_table

generate_figures = True  # only computes table
save_fig = True
test_run = False  # if true stops after first plot
recompute_cache = True

evaluations_dir = Path("./plots")

#api = "openai"
api = "openai_textEmbAda002"
dataset_b = f"causal_concepts"
#dataset_b_indices = f"causal_concepts_10000"
dataset_b_indices = None
template_set = statement_templates
prediction_cache_path = Path("./cache/")
base_path = Path("../queries")


templates_by_template_str = {
    **{t["pattern"]: t["name"] for t in question_templates},
    **{t["pattern"]: t["name"] for t in statement_templates},
}
assert len(question_templates) == len(statement_templates)

from_apis = [api] #["openai", "aleph_alpha", "opt"]
#prediction_names = ["proto_prediction", "NN_prediction"]
#prediction_labels = ["Prototype", "kNN"]
prediction_names = ["NN_prediction"]
prediction_labels = ["kNN"]

datasets = ["altitude", "causal_health", "driving", "recovery", "cancer", "earthquake"]
dataset_labels = ["Alitude", "Causal Health", "Driving", "Recovery", "Cancer", "Earthquake"]
datasets_a = datasets.copy()

figure_format = "pdf"

api_label = {
    "openai": "GPT-3",
    "openai_textEmbAda002": "Ada Text Emb 002"
}

if dataset_b_indices is None:
    indices_str = ""
else:
    indices_str = f"_{dataset_b_indices}"


def adj_mat_from_connections(var_names, connections):
    adj_mat = np.zeros((len(var_names), len(var_names)))
    for i, v0 in enumerate(var_names):
        for j, v1 in enumerate(var_names):
            if i == j:
                continue
            adj_mat[i, j] = connections[f"{v0}__{v1}"]
    return adj_mat


def load_and_compile_all_adj_mats(dataset_name, prediction_name):
    adj_mat = None
    for i, query_filter in enumerate(single_template_statement_filters):
        filter_name = query_filter["name"]
        temp_adj_mat, var_names_unique = load_single_adj_mat(dataset_name, prediction_name, filter_name)

        if adj_mat is None:
            adj_mat = np.empty_like(temp_adj_mat)

        # we only write the part where query template filter matches the question template.
        # we "discard" all non matching entries and get a single adj_mat that contains
        # results for matching query and question template.
        adj_mat[0, i, :, :] = temp_adj_mat[0, i, :, :]

    return adj_mat, var_names_unique


def load_single_adj_mat(dataset_name, prediction_name, filter_name):
    dataset_path_a = base_path / f"{api}_{dataset_name}_questions"

    dataset_embeddings_a, metas_a = load_embeddings(dataset_path_a)  # openai embedding size: 12288

    with open(base_path / f"{dataset_name}_full.pkl", "rb") as f:
        var_info_a = pickle.load(f)

    data_config_name = f"{dataset_name}"
    proto_config_name = f"{dataset_b}{indices_str}"
    file_name = f"{api}_{filter_name}_{data_config_name}_{proto_config_name}"
    with (prediction_cache_path / prediction_name / f"{file_name}.pkl").open("rb") as f:
        predictions = pickle.load(f)["prediction_bool"]

    current_template = None
    connections = {}
    var_names = []
    temp_adj_mat = None
    template_idx = 0

    for i, (meta, var_info, prediction) in enumerate(zip(metas_a, var_info_a, predictions)):
        template_name = templates_by_template_str[var_info["info"]['template']]
        v0 = var_info["info"]['names'][0]
        v1 = var_info["info"]['names'][1]
        var_names.append(v0)
        var_names.append(v1)

        if current_template is None:
            current_template = template_name
        if template_name != current_template:
            if temp_adj_mat is None:
                var_names_unique = sorted(list(set(var_names)))
                temp_adj_mat = np.zeros((1, len(template_set), len(var_names_unique),
                                         len(var_names_unique)))  # NUM_APIS, NUM_TEMPLATES, FROM_VAR, TO_VAR
            # plot(var_names, connections, dataset_a, current_template)

            temp_adj_mat[0, template_idx, :, :] = adj_mat_from_connections(var_names_unique, connections)
            current_template = template_name
            connections = {}
            var_names = []
            template_idx += 1

        connections[f"{v0}__{v1}"] = prediction
    # plot(var_names, connections, dataset_a, current_template)
    temp_adj_mat[0, template_idx, :, :] = adj_mat_from_connections(var_names_unique, connections)

    return temp_adj_mat, var_names_unique

pd_adj_mats = []

for prediction_name in prediction_names:
    d_adj_mats = []
    d_variable_names = []
    for dataset_idx, dataset in enumerate(datasets):
        adj_mats, variable_names = load_and_compile_all_adj_mats(dataset, prediction_name)
        d_adj_mats.append(adj_mats)
        d_variable_names.append(variable_names)  # fixme we create this multiple times but only use it once
    pd_adj_mats.append(d_adj_mats)

    """
    make_statistics_table(
            "stability_tables",  # "stability_table" - caching dir
            f"{api}_{prediction_name}",
            d_adj_mats,  # adj_mat = [DATATSETS][NUM_APIS, NUM_TEMPLATES, FROM_VAR, TO_VAR]
            api_labels=["GPT-3"], #, "Luminous", "OPT"],
            datasets=["altitude", "causal_health", "driving", "recovery", "cancer", "earthquake"],
            dataset_labels=["Altitude", "Health", "Driving", "Recovery", "Cancer", "Earthquake"],
            evaluations_dir=Path("./cache"),
            recompute_cache=recompute_cache,
            debug_recompute_metric=None  # prevents caching of results
    )
    """

    if not generate_figures:
        continue

    for dataset_idx, dataset in enumerate(datasets):
        adj_mats, variable_names = load_and_compile_all_adj_mats(dataset, prediction_name)

        for api_idx, api in enumerate(from_apis):
            rows = 1
            cols = len(template_set)

            subplot_size = cols
            fig, axs = plt.subplots(rows, cols, figsize=(subplot_size * cols, subplot_size))
            fig.suptitle(f'Model: {api_label[api]}; Dataset: {dataset.capitalize()}', y=1, fontsize=30)

            for template_idx, template in enumerate(template_set):
                ax = axs[template_idx]
                ax.set_xticks([], [])
                ax.set_yticks([], [])
                ax.set_aspect('equal')
                plt.sca(ax)
                ax.set_title(f'{template["name"]}', y=1, fontsize=25)

                adj_mat = adj_mats[api_idx, template_idx, :, :]
                adj_mat = adj_mat.clip(min=0.0)

                plot_from_adj_mat(adj_mat, variable_names, dataset, ax=ax, abrev_vars=True)

            graph_name = f"{dataset}"
            fig.tight_layout(h_pad=0.5)

            if save_fig:
                prediction_base_dir = evaluations_dir / prediction_name
                prediction_base_dir.mkdir(exist_ok=True, parents=True)
                plt.savefig(prediction_base_dir / f"{api}_{prediction_name}_{graph_name}.{figure_format}", dpi=200)
            else:
                plt.show()

            if test_run:
                exit()



cache_file = Path("../../evaluations/stability_table/statistics.pkl")

with cache_file.open("rb") as f:
    base_data = pickle.load(f)
pre_mean = base_data["mean"][:,0:1,:]  # pick GPT-3 results from base data
pre_std = base_data["std"][:,0:1,:]
pre_labels = ["Direct"]
pre_mean_std = (pre_mean, pre_std)
#pre_labels = []
#pre_mean_std = None


#[predictor][DATATSETS][NUM_APIS, NUM_TEMPLATES, FROM_VAR, TO_VAR]

comb_adj_mats = []
for i in range(len(pd_adj_mats[0])):
    comb_adj_mats.append(np.concatenate([d_adj_mat[i] for d_adj_mat in pd_adj_mats], axis=0))

make_statistics_table(
        "stability_table",  # "stability_table" - caching dir
        f"{api}_combined_predictions",
        comb_adj_mats,  # adj_mat = [DATATSETS][NUM_APIS, NUM_TEMPLATES, FROM_VAR, TO_VAR]
        d_variable_names,
        pre_results=pre_mean_std,
        api_labels=[*pre_labels, *prediction_labels], #["GPT-3"], #, "Luminous", "OPT"],
        datasets=["altitude", "causal_health", "driving", "recovery", "cancer", "earthquake"],
        dataset_labels=["Altitude", "Health", "Driving", "Recovery", "Cancer", "Earthquake"],
        evaluations_dir=Path("./cache"),
        recompute_cache=recompute_cache,
        debug_recompute_metric=None  # prevents caching of results
)
