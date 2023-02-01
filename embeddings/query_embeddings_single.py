import pickle
from pathlib import Path

from causalFM.api.api_helpers import lm_apis
from causalFM.embeddings.evaluations.helpers.embedding_processing import load_indices

dry_run = False
skip_existing = True  # skip samples for which a file exists
test_single = False  # if true, stops after the first query

def prepare_query_content(query_content):
    query_content = [q for q in query_content if q != "" and not q.startswith("#")]
    return query_content


#active_apis = ["openai", "aleph_alpha", "opt"]
#active_apis = ["openai"]
active_apis = ["openai_textEmbAda002"]
#datasets = ["frequent_nouns", ("frequent_words_5000", "frequent_words"), ("wordnet_nouns_100", "wordnet_nouns")]

list_datasets = [
    "question_answer",
    "answer_statements_questions"
]
statement_datasets = [
    "altitude_questions",
    "cancer_questions",
    "causal_chains_questions",
    "causal_health__alt_age_aging_questions",
    "causal_health__alt_health_conditions_questions",
    "causal_health__alt_health_healthiness_questions",
    "causal_health__alt_mobility_agility_questions",
    "causal_health__alt_mobility_fitness_questions",
    "causal_health__alt_nutrition_diet_questions",
    "causal_health__alt_nutrition_habits_questions",
    "causal_health_questions",
    "driving_questions",
    "earthquake_questions",
    "recovery_questions",
    "causal_statements_questions",
    "question_answer",
    "statement_wrongReasonsAltitude100_questions"
]
question_datasets = [
    "question_altitude_questions",
    "question_cancer_questions",
    "question_causal_health__alt_age_aging_questions",
    "question_causal_health__alt_health_conditions_questions",
    "question_causal_health__alt_health_healthiness_questions",
    "question_causal_health__alt_mobility_agility_questions",
    "question_causal_health__alt_mobility_fitness_questions",
    "question_causal_health__alt_nutrition_diet_questions",
    "question_causal_health__alt_nutrition_habits_questions",
    "question_causal_health_questions",
    "question_causal_statements_questions",
    "question_driving_questions",
    "question_earthquake_questions",
    "question_recovery_questions",
    "question_wrongReasonsAltitude100_questions"
]

datasets = [
    #*list_datasets
    #*statement_datasets
    #*question_datasets
    #"statement_wrongReasonsAltitude100_questions",
    #"question_wrongReasonsAltitude100_questions",
    #"statement_correctCausalReasons_questions",
    #"question_correctCausalReasons_questions"
    #("causal_concepts_questions", None, "causal_concepts_10000")
    "causal_concepts_questions"
]

keys_dir = Path("../keys")
queries_path = Path("./queries")


def main():
    for api_name in active_apis:
        print(f"[{api_name}] Starting up API.")

        query_count = 0
        answer_count = 0
        api = lm_apis[api_name]

        # TODO provide key as str
        model = api["model"](keys_dir, dry_run=dry_run)

        print(f"[{api_name}] Start querying.")
        for dataset_name in datasets:
            index_source = None
            if isinstance(dataset_name, str):
                source_file = dataset_name
            else:
                if len(dataset_name) == 2:
                    source_file, dataset_name = dataset_name
                elif len(dataset_name) == 3:
                    source_file, dataset_name, index_source = dataset_name
                    if dataset_name is None:
                        dataset_name = source_file
                else:
                    raise RuntimeError("unknown format")

            queries_list_file = queries_path / f"{source_file}.txt"
            with open(queries_list_file) as f:
                embedding_queries = prepare_query_content(f.read().splitlines())

            answer_dir = queries_path / f"{api_name}_{dataset_name}"
            answer_dir.mkdir(exist_ok=True)

            if index_source is None:
                index_list = range(len(embedding_queries))
            else:
                index_list = load_indices(queries_path / f"{index_source}.idcs")

            for i in index_list:
                query_text = embedding_queries[i]
                query_count += 1

                answer_path = answer_dir / f"{i}_embedding.pkl"
                if skip_existing and answer_path.exists():
                    print(f"[Query {i}]: File exists. Skipping.")
                    continue

                embedding = model.query_embedding(query_text, log_info=i)

                if embedding is None:
                    print(f"[Query {i}] No response.")
                    continue
                answer_count += 1

                info = {
                    "api": api_name,
                    "dataset": dataset_name,
                    "query_text": query_text,
                    "embedding": embedding
                }
                with answer_path.open("wb+") as f:
                    pickle.dump(info, f)

                if test_single:
                    break
            if test_single:
                break
        if test_single:
            break

        print(f"Found {query_count} queries.")
        print(f"Recorded {answer_count} new answers.")


if __name__ == "__main__":
    main()
