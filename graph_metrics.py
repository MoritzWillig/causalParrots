import numpy as np
import cdt

from causalFM.query_helpers import question_templates


def precision(gt, pred):
    true_positives = np.sum(np.maximum(pred - 1 + gt, 0))
    pred_positives = np.sum(pred)
    if pred_positives == 0:
        return 0
    else:
        return true_positives / pred_positives


def recall(gt, pred):
    true_positives = np.sum(np.maximum(pred - 1 + gt, 0))
    all_positives = np.sum(gt)
    return true_positives / all_positives


def f1_score(gt, pred):
    prec = precision(gt, pred)
    rec = recall(gt, pred)
    if prec+rec == 0.0:
        f1 = 0
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return f1


def SID(gt, pred):
    return cdt.metrics.SID(gt, pred)


def sparsity(gt, pred):
    size = len(gt)
    max_num_edges = size * (size - 1)
    num_edges = np.sum(pred)
    return 1.0 - (num_edges / max_num_edges)


def decisiveness(gt, pred):
    size = len(pred)

    sym = 0
    asym = 0
    for i in range(size):
        for j in range(i):  # does not check main diagonal.
            a = pred[i, j]
            b = pred[j, i]
            if a == b:
                if a == 1:
                    sym += 1
            else:
                asym += 1

    n_edges = asym+sym
    if n_edges == 0:
        return 0.0
    else:
        return asym/(asym+sym)


def decisiveness_delta_conv(result, directedness=[template["direction"] == "directed" for template in question_templates]):
    # FIXME default values for directedness argument assume default template ordering

    asym_idx = [i for i, d in enumerate(directedness) if d]
    sym_idx = [i for i, d in enumerate(directedness) if not d]

    asym_results = result[:, :, asym_idx]
    sym_results = result[:, :, sym_idx]

    asym_results_mean = np.mean(asym_results, axis=-1)
    sym_results_mean = np.mean(sym_results, axis=-1)

    #asym_results_std = np.mean(sym_results, axis=-1)
    #sym_results_std = np.mean(sym_results, axis=-1)

    avg_delta = asym_results_mean - sym_results_mean

    return avg_delta, None



#metric_names = ["SID", "SHD", "Precision", "Recall", "Sparsity", "Decisiveness"]
#metric_higher_is_better = [False, False, True, True, None, True]
"""metric_funcs = [
    cdt.metrics.SID,
    cdt.metrics.SHD,
    precision,
    recall,
    sparsity,
    decisiveness
]"""

graph_metrics = [
    {
        "name": "SID",
        "higher_is_better": False,
        "func": cdt.metrics.SID
    },
    {
        "name": "SHD",
        "higher_is_better": False,
        "func": cdt.metrics.SHD
    },
    #{
    #    "name": "Precision",
    #    "higher_is_better": True,
    #    "func": precision
    #},
    #{
    #    "name": "Recall",
    #    "higher_is_better": True,
    #    "func": recall
    #},
    {
        "name": "F1 Score",
        "higher_is_better": True,
        "func": f1_score
    },
    {
        "name": "Sparsity",
        "higher_is_better": None,
        "func": sparsity
    },
    #{
    #    "name": "Decisiveness",
    #    "higher_is_better": True,
    #    "func": decisiveness
    #},
    {
        "name": "Decisiveness Delta",
        "latex": "$\\Delta\\text{Decisiveness}_\\text{Sym,Asym}$",
        "higher_is_better": True,
        "func": decisiveness,
        "conv_func": decisiveness_delta_conv,
        "computes_std": False
    }
]
