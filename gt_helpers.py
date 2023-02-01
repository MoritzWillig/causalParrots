import numpy as np
import networkx as nx

from causalFM.generate_altitude import variables as altitude_vars
from causalFM.generate_cancer import variables as cancer_vars
from causalFM.generate_causal_health import variables as causal_health_vars
from causalFM.generate_driving import variables as driving_vars
from causalFM.generate_earthquake import variables as earthquake_vars
from causalFM.generate_recovery import variables as recovery_vars


def get_var_names(vars):
    return [var.name for var in vars]


dataset_ground_truth = {
    "altitude": {
        "vars": get_var_names(altitude_vars),
        "edges": [
            ["altitude", "temperature"]
        ]
    },
    "cancer": {
        "vars": get_var_names(cancer_vars),
        "edges": [
            ["pollution", "cancer"],
            ["smoking", "cancer"],
            ["cancer", "x-ray"],
            ["cancer", "dyspnoea"]
        ]
    },
    "causal_health": {
        "vars": get_var_names(causal_health_vars),
        "edges": [
            ["age", "nutrition"],
            ["age", "health"],
            ["nutrition", "health"],
            ["health", "mobility"],
        ]
    },
    "driving": {
        "vars": get_var_names(driving_vars),
        "edges": [
            ["driveStyle", "fuel"],
            ["carType", "fuel"]
        ]
    },
    "earthquake": {
        "vars": get_var_names(earthquake_vars),
        "edges": [
            ["burglaries", "alarms"],
            ["earthquake", "alarms"],
            ["alarms", "john"],
            ["alarms", "mary"],
        ]
    },
    "recovery": {
        "vars": get_var_names(recovery_vars),
        "edges": [
            ["treatment", "recovery"],
            ["precondition", "recovery"]
        ]
    },
}


#def get_dataset_var_names(dataset_name):
#    return dataset_ground_truth[dataset_name]["vars"].copy()


def get_dataset_edges(dataset_name):
    return dataset_ground_truth[dataset_name]["edges"].copy()


#def get_gt_adjacency(dataset_name, variable_names):
#    #vars = dataset_ground_truth[dataset_name]["vars"]
#    vars = variable_names
#    edges = dataset_ground_truth[dataset_name]["edges"]
#
#    adj = np.zeros((len(vars), len(vars)))
#    for e in edges:
#        e_from = e[0]
#        e_to = e[1]
#        adj[e_from][e_to] = 1
#    return adj


def get_gt_graph(dataset_name, variable_names):
    graph = nx.DiGraph()

    #node_names = get_dataset_var_names(dataset_name)
    node_names = variable_names
    for node_name in node_names:
        graph.add_node(node_name)

    edges = get_dataset_edges(dataset_name)
    for cause, effect in edges:
        graph.add_edge(cause, effect)

    return graph


def get_graph_from_adj_mat(adj_mat, node_names):
    num_vars = len(node_names)
    if adj_mat.shape[0] != adj_mat.shape[1]:
        raise ValueError("Adjacency matrix has to be a square matrix.")
    if adj_mat.shape[0] != num_vars:
        raise ValueError("Adjacency matrix size does not match node names.")

    graph = nx.DiGraph()

    for node_name in node_names:
        graph.add_node(node_name)

    for from_idx in range(num_vars):
        for to_idx in range(num_vars):
            if adj_mat[from_idx, to_idx]:
                graph.add_edge(node_names[from_idx], node_names[to_idx])

    return graph
