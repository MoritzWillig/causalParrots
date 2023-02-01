from pathlib import Path
import time

from causalFM.api.aleph_alpha_api import startup_aleph_alpha, query_aleph_alpha
from causalFM.api.openai_api import startup_openai, query_openai
from causalFM.api.opt_api import startup_opt, query_opt
from causalFM.query_helpers import load_query_instances

dry_run = False  # does not query the APIs. (But saves results, if dry run returns something else than None).
skip_existing = True  # skip samples for which a file exists
test_single = False  # if true, stops after the first query

active_apis = ["openai", "aleph_alpha", "opt"]
#active_apis = ["openai", "aleph_alpha"]
#active_apis = ["openai"]
#active_apis = ["aleph_alpha"]
#active_apis = ["opt"]
#datasets = ["altitude", "causal_health", "driving", "recovery", "cancer", "earthquake", "intuitive_physics", "causal_chain"] # base
#datasets = [
#    "causal_health__alt_age_aging",
#    "causal_health__alt_health_conditions",
#    "causal_health__alt_health_healthiness",
#    "causal_health__alt_mobility_agility",
#    "causal_health__alt_mobility_fitness",
#    "causal_health__alt_nutrition_diet",
#    "causal_health__alt_nutrition_habits"
#] # alternatives
#datasets = ["altitude", "causal_health", "driving", "recovery", "cancer", "earthquake", "intuitive_physics", "causal_chain",
#    "causal_health__alt_age_aging",
#    "causal_health__alt_health_conditions",
#    "causal_health__alt_health_healthiness",
#    "causal_health__alt_mobility_agility",
#    "causal_health__alt_mobility_fitness",
#    "causal_health__alt_nutrition_diet",
#    "causal_health__alt_nutrition_habits"
#]
datasets = ["causal_chains"]

keys_dir = Path("./keys")
queries_path = Path("./queries")

apis = {
    "openai": {
        "startup": startup_openai,
        "query": query_openai,
        "limit": 5  # max requests per min
    },
    "aleph_alpha": {
        "startup": startup_aleph_alpha,
        "query": query_aleph_alpha,
        "limit": None  # max requests per min
    },
    "opt": {
        "startup": startup_opt,
        "query": query_opt,
        "limit": None  # max requests per min
    }
}


def main():
    for api_name in active_apis:
        print(f"[{api_name}] Starting up API.")

        query_count = 0
        answer_count = 0
        api = apis[api_name]
        context = api["startup"](keys_dir)
        query_func = api["query"]

        print(f"[{api_name}] Start querying.")
        for dataset_name in datasets:
            questions = load_query_instances(queries_path / f"{dataset_name}_questions.txt")

            answer_dir = queries_path / f"{api_name}_{dataset_name}"
            answer_dir.mkdir(exist_ok=True)

            if api["limit"] is not None:
                min_request_time = 61.0 / api["limit"]
            else:
                min_request_time = None

            last_query_time = time.perf_counter()
            for i, question in enumerate(questions):
                query_count += 1

                answer_path = answer_dir / f"{i}.txt"
                if skip_existing and answer_path.exists():
                    print(f"[Query {i}]: File exists. Skipping.")
                    continue

                response_text = query_func(context, question, dry_run=dry_run)

                if response_text is None:
                    print(f"[Query {i}] No response.")
                    continue
                with answer_path.open("w+") as f:
                    f.write(response_text)
                    answer_count += 1

                if test_single:
                    return

                if min_request_time is not None:
                    current_time = time.perf_counter()
                    elapsed_time = current_time - last_query_time
                    time.sleep(min_request_time - elapsed_time)
                    last_query_time = time.perf_counter()


        print(f"Observed {query_count} queries.")
        print(f"Recorded {answer_count} answers.")


if __name__ == "__main__":
    main()
