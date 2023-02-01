from causalFM.embeddings.evaluations.helpers.similarity_helpers import get_metric_func

from pathlib import Path
import pickle
import numpy as np

from causalFM.embeddings.evaluations.helpers.embedding_processing import load_embeddings, load_indices, \
    query_statement_filters

#apis = ["openai"]
apis = ["openai_textEmbAda002"]

datasets = ["altitude", "causal_health", "driving", "recovery", "cancer", "earthquake"]
datasets_a = datasets
dataset_b = f"causal_concepts"
#dataset_b_indices = f"causal_concepts_10000"
dataset_b_indices = None

base_path = Path("../queries")

prediction_cache_path = Path("./cache/NN_prediction")
prediction_cache_path.mkdir(parents=True, exist_ok=True)


if dataset_b_indices is None:
    indices_str = ""
else:
    indices_str = f"_{dataset_b_indices}"


def predict_from_NN(data, filtered_dataset_embeddings, metric=get_metric_func("cos_sim")):
    # FIXME assumes filtered_dataset_embeddings samples are alternatively labeled: true, false
    result = np.empty((len(data),))
    closest_meta = np.empty((len(data), 2))
    for i, embedding in enumerate(data):
        max_sim = -1
        max_sim_idx = -1
        label = -1
        for j, known_embedding in enumerate(filtered_dataset_embeddings):
            distance = metric(embedding, known_embedding)
            if distance > max_sim:  # FIXME this works form cos_sim and dot (where greater is better), but not for euclidean
                max_sim = distance
                max_sim_idx = j
                label = 1 - (j % 2)  # set to 1 if even index (true embedding), set to 0 if odd index (false embedding)
        result[i] = label
        closest_meta[i, 0] = max_sim
        closest_meta[i, 1] = max_sim_idx
    return result, closest_meta


def create_NN_predictor(filtered_dataset_embeddings, metric=get_metric_func("cos_sim")):
    return lambda data: predict_from_NN(data, filtered_dataset_embeddings, metric=metric)


for api in apis:
    print("loading NN embeddings")
    dataset_path_b = base_path / f"{api}_{dataset_b}_questions"
    if dataset_b_indices is not None:
        dataset_indices_path_b = base_path / f"{dataset_b_indices}.idcs"
        indices_b = load_indices(dataset_indices_path_b)
    else:
        indices_b = None
    dataset_embeddings_b, metas_b = load_embeddings(dataset_path_b, indices_path=indices_b)

    dataset_b_queries_path = base_path / f"{dataset_b}_full.pkl"
    with dataset_b_queries_path.open("rb") as f:
        queries = pickle.load(f)
        if indices_b is not None:
            queries = [queries[i] for i in indices_b]

    for query_filter in query_statement_filters:
        filter_name = query_filter["name"]
        filter_func = query_filter["func"]
        print(f"computing {filter_name}")

        if filter_func is None:
            filtered_dataset_embeddings = dataset_embeddings_b
        else:
            filtered_dataset_embeddings = np.array(filter_func(dataset_embeddings_b, metas_b, queries))

        predictor = create_NN_predictor(filtered_dataset_embeddings)

        for dataset in datasets:
            dataset_name_a = dataset
            print(f"processing embeddings: {dataset_name_a}")
            dataset_path_a = base_path / f"{api}_{dataset_name_a}_questions"

            dataset_embeddings_a, metas_a = load_embeddings(dataset_path_a)  # openai embedding size: 12288

            prediction_bool, prediction_closest_meta = predictor(dataset_embeddings_a)

            data_config_name = f"{dataset_name_a}"
            proto_config_name = f"{dataset_b}{indices_str}"
            file_name = f"{api}_{filter_name}_{data_config_name}_{proto_config_name}"
            with (prediction_cache_path / f"{file_name}.pkl").open("wb+") as f:
                pickle.dump({
                    "prediction_bool": prediction_bool,
                    "prediction_closest_meta": prediction_closest_meta
                }, f)
            with (prediction_cache_path / f"{file_name}.txt").open("w+") as f:
                f.write("\n".join(["1" if x == 1 else "0" for x in prediction_bool]))

print("done.")
