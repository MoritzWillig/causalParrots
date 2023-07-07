from pathlib import Path
from causalFM.query_helpers import load_query_instances

dry_run = False

keep_existing_summaries = True
append_to_incomplete_summaries = True  # This only makes sence with 'keep_existing_summaries=True', otherwise all summaries are recreated anyway
extended_auto_decision = True  # auto classify other answers only 'yes' and 'no'

skip_missing_combinations = True  # otherwise crashes if files are missing

#from_apis = ["openai", "aleph_alpha", "opt"]
from_apis = ["gpt_4"]
#from_apis = ["aleph_alpha"]
#from_apis = ["openai"]
datasets = ["altitude", "causal_health", "driving", "recovery", "cancer", "earthquake",
            "causal_health__alt_age_aging",
            "causal_health__alt_health_conditions",
            "causal_health__alt_health_healthiness",
            "causal_health__alt_mobility_agility",
            "causal_health__alt_mobility_fitness",
            "causal_health__alt_nutrition_diet",
            "causal_health__alt_nutrition_habits"
            ]

for api_name in from_apis:
    for dataset_name in datasets:
        answer_dir = Path(f"./queries/{api_name}_{dataset_name}")
        if skip_missing_combinations and not answer_dir.exists():
            print(f"[{api_name}_{dataset_name}] Answers missing. Skip.")
            continue

        queries = load_query_instances(Path(f"./queries/{dataset_name}_questions.txt"))
        decisions = []

        summary_path = answer_dir / "summary.txt"
        if keep_existing_summaries and summary_path.exists():
            # read existing summary and check if entries are missing
            with summary_path.open("r") as f:
                decisions = [d.strip() for d in f.readlines()]
            if not append_to_incomplete_summaries or len(queries) == len(decisions):
                print(f"[{api_name}_{dataset_name}] Summary exists. Skipping")
                continue

        print(f"[{api_name}_{dataset_name}] {len(queries)} queries")
        for i, query in enumerate(queries):
            if i < len(decisions):
                print(f"[decision {i}] Existing. Skipping.")
                continue

            answer = Path(answer_dir / f"{i}.txt").read_text()
            print("===QUERY===")
            print(query)
            print("===ANSWER===")
            print(answer.strip())

            auto_answer = answer.strip().lower()
            auto_answer_lines = auto_answer.split("\n")
            num_auto_answer_lines = len(auto_answer_lines)

            if extended_auto_decision and num_auto_answer_lines > 1 and auto_answer_lines[1].startswith("b:"):
                # output is in quiz format A:... B:... C:...

                # check first answer
                if auto_answer_lines[0].startswith("yes"):
                    print("[auto QUIZ (y)]")
                    decisions.append("uqy")
                elif auto_answer_lines[0].startswith("no"):
                    print("[auto QUIZ (n)]")
                    decisions.append("uqn")
                else:
                    print("[auto QUIZ]")
                    decisions.append("uq")
            elif (extended_auto_decision and auto_answer.startswith("no")) or auto_answer == "no":
                print("[auto NO]")
                if num_auto_answer_lines > 1 and "why?" in auto_answer_lines[1]:
                    #detect follow up "Why? question and answer"
                    decisions.append("ne")
                else:
                    decisions.append("n")
            elif (extended_auto_decision and auto_answer.startswith("yes")) or auto_answer == "yes":
                print("[auto YES]")
                if num_auto_answer_lines > 1 and "why?" in auto_answer_lines[1]:
                    decisions.append("ye")
                else:
                    decisions.append("y")
            else:
                # manual classification
                print("===[Y|N|YP|NP|YO|NO|YI|NI|YE|NE|WD|UQ|UQY|UQN|U|X|META]===")
                decision = None

                # yp="yes, probably", yi="yes, indirectly", yo="yes, also other factors", ye="yes + explanation",
                # u="undecided/inconclusive", x=no answer (like a general statement)
                # uq=output is in quiz format (A:... B:... C:...) uqy=first quiz answer is correct.
                # wd="causality was interpreted in the wrong direction"
                # meta="As an AI, I don't have specific information ...", "The text doesn't provide enough information to determine ...",
                #      "As an AI, I don't have access to personal data about individuals ..."
                while decision not in ["y", "n", "yp", "np", "yi", "ni", "yo", "no", "ye", "ne", "u", "uq", "uqy", "uqn", "wd", "x", "meta"]:
                    decision = input().strip().lower()
                decisions.append(decision)

        if not dry_run:
            with summary_path.open("w+") as f:
                for decision in decisions:
                    f.write(decision+"\n")
