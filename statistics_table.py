import pickle
from pathlib import Path

import networkx as nx
import numpy as np

from causalFM.answer_helpers import get_response_flags, load_compact_answers
from causalFM.graph_metrics import graph_metrics
from causalFM.query_helpers import question_templates
from causalFM.gt_helpers import get_gt_graph, get_graph_from_adj_mat

#api_names = ["GPT-3", "Luminous", "OPT"]
#datasets = ["altitude", "causal_health", "driving", "recovery", "cancer", "earthquake"]
#dataset_labels = ["Altitude", "Health", "Driving", "Recovery", "Cancer", "Earthquake"]

#allow_quiz_answers = True  # include quiz-style answers
#positive_response_flags, negative_response_flags, undecided_response_flags = get_response_flags(allow_quiz_answers)
# d_adj_mats = []  # adj_mat = [DATATSETS][NUM_APIS, NUM_TEMPLATES, FROM_VAR, TO_VAR]
# for dataset in datasets:
#    adj_mats, variable_names, _ = load_compact_answers(dataset, from_apis, len(question_templates), positive_response_flags, negative_response_flags, undecided_response_flags)
#    d_adj_mats.append(adj_mats)

def compute_cache(
        d_adj_mats,  # adj_mat = [DATATSETS][NUM_APIS, NUM_TEMPLATES, FROM_VAR, TO_VAR]
        d_variable_names,
        datasets,
        base_name="stability_tables",
        exp_name="base",
        evaluations_dir=Path("./evaluations"),
        recompute_cache=False,
        debug_recompute_metric=None  # prevents caching of results
    ):

    base_dir = evaluations_dir / base_name
    if not debug_recompute_metric:
        base_dir.mkdir(exist_ok=True)


    cache_file = base_dir / f"{exp_name}_statistics.pkl"

    num_apis = d_adj_mats[0].shape[0]


    if recompute_cache and debug_recompute_metric is not None:
        print("debug_recompute_metric: debug evaluation of metric - NO CACHING OF RESULTS")
        #metric_names = [metric_names[i] for i in debug_recompute_metric]
        #metric_higher_is_better = [metric_higher_is_better[i] for i in debug_recompute_metric]
        global graph_metrics
        graph_metrics = [graph_metrics[i] for i in debug_recompute_metric]

    results = [np.empty((num_apis, len(datasets), len(question_templates))) for _ in graph_metrics]

    if recompute_cache or not cache_file.exists():
        for dataset_idx, dataset in enumerate(datasets):
            adj_mats = d_adj_mats[dataset_idx]  # [api_idx, :, :, :]

            #node_names = get_dataset_var_names(dataset)
            node_names = d_variable_names[dataset_idx]

            gt_graph = get_gt_graph(dataset, node_names)
            gt_graph = np.array(nx.adjacency_matrix(gt_graph, gt_graph.nodes()).todense(), dtype=int)

            for api_idx in range(num_apis):
                for template_idx, template in enumerate(question_templates):
                    #directed = template["direction"] == "directed"

                    adj_mat = adj_mats[api_idx, template_idx, :, :]
                    adj_mat = adj_mat.clip(min=0.0)  # 0=No edge, 1=Edge, -1=Undecided->No Edge

                    predicted_graph = get_graph_from_adj_mat(adj_mat, node_names)
                    predicted_graph = np.array(nx.adjacency_matrix(predicted_graph, predicted_graph.nodes()).todense(), dtype=int)

                    for res, metric in zip(results, graph_metrics):
                        res[api_idx, dataset_idx, template_idx] = metric["func"](gt_graph, predicted_graph)

        def mean_conv(result):
            return np.mean(result, axis=-1), np.std(result, axis=-1)

        results_mean = []
        results_std = []
        for result, metric in zip(results, graph_metrics):
            conv_func = metric.get("conv_func", mean_conv)
            mean, std = conv_func(result)
            results_mean.append(mean)
            if std is None:
                std = np.zeros((num_apis, len(datasets)))
            results_std.append(std)
        results_mean = np.array(results_mean)
        results_std = np.array(results_std)

        if debug_recompute_metric:
            print("debug_recompute_metric: results")
            print(results_mean)
            print(results_std)
            exit()

        with cache_file.open("wb+") as f:
            pickle.dump({
                "mean": results_mean,
                "std": results_std
            }, f)
    else:
        with cache_file.open("rb+") as f:
            file_data = pickle.load(f)
            results_mean = file_data["mean"]
            results_std = file_data["std"]

    print(results_mean)
    print(results_std)
    print("ok")

    return results_mean, results_std


def make_table_content(
        results_mean, results_std,
        metric_indices,
        api_labels=["GPT-3", "Luminous", "OPT"],
        dataset_labels=["Altitude", "Health", "Driving", "Recovery", "Cancer", "Earthquake"],
        make_begin=True, make_end=True, include_dataset_names=True):
    if isinstance(metric_indices, int):
        metric_indices = [metric_indices]

    show_box = False
    show_content_lines = False
    content_hline = "\\hline" if show_content_lines else ""
    content_vline = "|" if show_content_lines else " "
    outer_hline = "\\hline" if show_box else ""
    outer_vline = "|" if show_box else ""
    delimiter_line = "\\hline"  # string appended if make_end is False

    direction_strs = []
    for m_idx in range(len(graph_metrics)):
        if graph_metrics[m_idx]["higher_is_better"] is None:
            direction_strs.append(f"")
        elif graph_metrics[m_idx]["higher_is_better"]:
            direction_strs.append(f" $\\uparrow$")
        else:
            direction_strs.append(f" $\\downarrow$")

    metric_names = []
    for metric in graph_metrics:
        metric_names.append(metric.get('latex', metric['name']))

    h1 = "{} & " + " & ".join([f"\multicolumn{{{len(dataset_labels)}}}{{c}}{{{metric_names[m_idx]}{direction_strs[m_idx]}}}" for m_idx in metric_indices])
    if include_dataset_names:
        h2 = "{} & " + " & ".join(dataset_labels * len(metric_indices)) + "\\\\\n"
    else:
        h2 = ""
    content = []

    #best_idcs = np.amax(results_mean, axis=1)
    best_vals = np.empty((len(metric_indices), len(dataset_labels)))
    for i, m_idx in enumerate(metric_indices):
        if graph_metrics[m_idx]["higher_is_better"] is None:
            best_vals[i, ...] = np.NaN
        elif graph_metrics[m_idx]["higher_is_better"]:
            best_vals[i, ...] = np.max(results_mean[m_idx, ...], axis=0)
        else:
            best_vals[i, ...] = np.min(results_mean[m_idx, ...], axis=0)
    val_eps = 0.01

    for api_idx, api_name in enumerate(api_labels):
        temp = [f"{api_name}"]
        for me_idx, metric_idx in enumerate(metric_indices):
            for dataset_idx in range(len(dataset_labels)):
                val_m = results_mean[metric_idx, api_idx, dataset_idx]
                val_s = results_std[metric_idx, api_idx, dataset_idx]

                if abs(best_vals[me_idx, dataset_idx] - val_m) < val_eps:
                    val_m_str = f"\\mathbf{{{val_m:.2f}}}"
                else:
                    val_m_str = f"{val_m:.2f}"

                std_str = f"_{{\\pm{val_s:.2f}}}" if graph_metrics[metric_idx].get('computes_std', True) else ""

                cell = f"${val_m_str}{std_str}$"
                #temp.append(f"{val_m:.2f} $\\pm$ {val_s:.2f}")
                #temp.append(f"{val_m:.2f}")
                temp.append(cell)

        content.append("&".join(temp))

    content_line_end = f"\\\\ {content_hline} \n"
    if make_begin:
        table = f'\\begin{{tabular}}{{{outer_vline}r|{content_vline.join(["l"] * (len(dataset_labels) * len(metric_indices)))}{outer_vline}}} {outer_hline}\n'
    else:
        table = ""
    table += f"""{h1} \\\\
{h2}{content_line_end.join(content)} \\\\{outer_hline if make_end else ""}
"""
    if make_end:
        table += f"\\end{{tabular}}"
    else:
        table += f"{delimiter_line}\n"
    return table


def make_statistics_table(
        base_name,  # "stability_table" - caching dir
        exp_name,
        d_adj_mats,  # adj_mat = [DATATSETS][NUM_APIS, NUM_TEMPLATES, FROM_VAR, TO_VAR]
        d_variable_names,
        api_labels=["GPT-3", "Luminous", "OPT"],
        datasets=["altitude", "causal_health", "driving", "recovery", "cancer", "earthquake"],
        dataset_labels=["Altitude", "Health", "Driving", "Recovery", "Cancer", "Earthquake"],
        evaluations_dir=Path("./evaluations"),
        recompute_cache=False,
        debug_recompute_metric=None,  # prevents caching of results
        pre_results=None,  # cached (mean,std) to prepend to other data
):
    results_mean, results_std = compute_cache(
        d_adj_mats,
        d_variable_names,
        datasets,
        base_name,
        exp_name,
        evaluations_dir,
        recompute_cache,
        debug_recompute_metric
    )
    if pre_results is not None:
        pre_mean, pre_std = pre_results
        results_mean = np.concatenate([pre_mean, results_mean], axis=1)
        results_std = np.concatenate([pre_std, results_std], axis=1)

    num_metrics = len(graph_metrics)
    table = ""
    for i in range(num_metrics):
        table += make_table_content(
            results_mean, results_std,
            metric_indices=[i],
            api_labels=api_labels,
            dataset_labels=dataset_labels,
            make_begin=i == 0, make_end=i == num_metrics-1, include_dataset_names=i == 0)

    print(f"------{base_name}--------")
    print(table)
