import pickle
import numpy as np
from pathlib import Path
from causalFM.embeddings.evaluations.helpers.similarity_helpers import get_metric_func
from causalFM.query_helpers import statement_templates


def load_indices(index_list_path: Path):
    with index_list_path.open("r") as f:
        content = f.read()
    index_list = [int(s) for s in content.split(",") if not s.isspace() and s != ""]
    return index_list


def load_embeddings(directory_path: Path, indices_path=None):
    """
    :param path:
    :return: [N, embedding_size] np-array
    """

    embeddings = []
    metas = []

    if indices_path is None:
        indices = None
    else:
        if isinstance(indices_path, Path):
            indices = load_indices(indices_path)
        else:
            indices = indices_path

    if not directory_path.exists():
        raise RuntimeError(f"path does not exist: {directory_path}")

    if indices is None:
        i = 0
        #for file_path in sorted(list(directory_path.iterdir())): sorting puts "10" before "2", "3", ...
        try:
            while True:
                file_path = directory_path / f"{i}_embedding.pkl"
                with file_path.open("rb") as f:
                    file_obj = pickle.load(f)
                    #"query_text"
                    file_obj["index"] = i
                    embedding = file_obj["embedding"]
                    embeddings.append(embedding)
                    metas.append(file_obj)
                i += 1
        except FileNotFoundError:
            pass
    else:
        for i in indices:
            file_path = directory_path / f"{i}_embedding.pkl"
            with file_path.open("rb") as f:
                file_obj = pickle.load(f)
                # "query_text"
                file_obj["index"] = i
                embedding = file_obj["embedding"]
                embeddings.append(embedding)
                metas.append(file_obj)

    #print(f"loaded {len(embeddings)} embeddings")
    return np.array(embeddings), metas


def compute_cosine_matrix(embeddings_a, embeddings_b=None, metric=get_metric_func("cos_sim"), is_symetric=True):
    symetric = is_symetric
    if embeddings_b is None:
        embeddings_b = embeddings_a
    if is_symetric:
        assert embeddings_a.shape == embeddings_b.shape
    num_a = len(embeddings_a)
    num_b = len(embeddings_b)
    matrix = np.ones((num_a, num_b))
    for i in range(num_a):
        a = embeddings_a[i, :]

        start = i + 1 if symetric else 0
        for j in range(start, num_b):
            b = embeddings_b[j, :]
            cos_sim = metric(a, b)
            matrix[i, j] = cos_sim
            if symetric:
                matrix[j, i] = cos_sim
    return matrix


def compute_average_similarity(sim_matrix: np.ndarray, exclude_main_diagonal=False):
    if exclude_main_diagonal:
        assert sim_matrix.shape[0] == sim_matrix.shape[1]
        sim_flat = sim_matrix[np.tril_indices(sim_matrix.shape[0], k=-1)] # get all entries of lower triangular matrix (without the diagonal)
        mean = np.mean(sim_flat)
    else:
        mean = np.mean(sim_matrix)
    return mean



statement_templates_directed = {s["pattern"]: s["direction"] == "directed" for s in statement_templates}
statement_templates_id = {s["pattern"]: i for i, s in enumerate(statement_templates)}


def filter_asym(embeddings, metas, queries, templates=statement_templates_directed):
    return [embedding for embedding, meta, query in zip(embeddings, metas, queries) if [query["info"]["template"]]]


def filter_sym(embeddings, metas, queries, templates=statement_templates_directed):
    return [embedding for embedding, meta, query in zip(embeddings, metas, queries) if not templates[query["info"]["template"]]]


def generate_filter_by_id(template_id, templates_by_id=statement_templates_id):
    return lambda embeddings, metas, queries: [embedding for embedding, query in zip(embeddings, queries) if templates_by_id[query["info"]["template"]] == template_id]


single_template_statement_filters = [{
    "name": f"template{i}",
    "func": generate_filter_by_id(i)
} for i in range(len(statement_templates))]

query_statement_filters = [
    {
        "name": "all_templates",
        "func": None
    },
    {
        "name": "asym",
        "func": filter_asym
    },
    {
        "name": "sym",
        "func": filter_sym
    },
    *single_template_statement_filters
]


def predict_from_prototypes(data, yes_proto, no_proto, metric=get_metric_func("cos_sim")):
    result = np.empty((len(data),))
    for i, embedding in enumerate(data):
        yes_cosine = metric(embedding, yes_proto)
        no_cosine = metric(embedding, no_proto)
        result[i] = yes_cosine - no_cosine
    return np.where(result > 0, 1, 0), result  # FIXME this works form cos_sim and dot (greater is better), but not for euclidean


def create_embedding_predictor(yes_embedding, no_embedding, metric=get_metric_func("cos_sim")):
    return lambda data: predict_from_prototypes(data, yes_embedding, no_embedding, metric=metric)
