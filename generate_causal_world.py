from causalFM.query_helpers import store_query_instances

dry_run = False
queries_path = "./queries/causal_world"

cot_cond_texts = [
"""Q: If flipping switches causes light bulbs to shine and shining light bulbs cause moths to appear. Does flipping switches causes moths to appear?
A: Because flipping switches causes light bulbs to shine and shining light bulbs causes moths to appear, flipping switches causes moths to appear. The answer is yes.""",

"""Q: If heavy rain causes the streets to flood and shining light bulbs cause moths to appear. Does heavy rain cause moths to appear?
A: Because heavy rain causes the streets to flood, but is not connected to shining light bulbs which allow for moths to appear. The answer is no.""",

"""Q: If Sibfan causes flooded streets and heavy rain causes Sibfan. Does heavy rain cause flooded streets?
A: Because heavy rain causes Sibfan and Sibfan causes flooded streets. The answer is yes.""",

"""Q: If Sibfan causes heavy rain and Sibfan causes light bulbs to shine. Does Sibfan cause heavy rain?
A: Because Sibfan directly causes heavy rain, the answer is yes.""",
]
cot_world_postfix = "\nA:"


def create_cot_world_prefix(n):
    cot_prefix = "\n\n".join(cot_cond_texts[:n]) + "\n\nQ: "
    return cot_prefix


causal_world_data = [
    ["general", "If rising temperatures cause the poles to melt and melting poles cause the sea level to rise. Do rising temperatures cause the sea level to rise?"],  # YES
    ["general", "If rising temperatures cause people to walk faster and fast walking people cause the sea level to rise. Do rising temperatures cause the sea level to rise?"],  # YES
    ["general", "If rising temperatures cause people to walk faster and melting poles cause the sea level to rise. Do rising temperatures cause the sea level to rise?"],  # NO
    ["general", "If fast walking people cause the poles to melt and melting poles cause the sea level to rise. Do fast walking people cause the sea level to rise?"],  # YES
    ["general", "If rising temperatures cause the poles to melt and melting poles cause people to walk faster. Do rising temperatures cause the sea level to rise?"],  # NO

    # rising temperatures -> Quab, melting poles -> Blaong, people walk faster -> Aiftan, rising sea level -> Wahrg
    ["im_rotation", "If Quab causes Blaong and Blaong causes Wahrg. Does Quab cause Wahrg?"],  # YES
    ["im_rotation", "If Quab causes Blaong and Blaong causes Wahrg. Does Wahrg cause Quab?"],  # NO
    ["im_rotation", "If Wahrg causes Quab and Quab causes Blaong. Does Wahrg cause Blaong?"],  # YES
    ["im_rotation", "If Wahrg causes Quab and Quab causes Blaong. Does Blaong cause Wahrg?"],  # NO
    ["im_rotation", "If Blaong causes Wahrg and Wahrg causes Quab. Does Blaong cause Quab?"],  # YES
    ["im_rotation", "If Blaong causes Wahrg and Wahrg causes Quab. Does Quab cause Blaong?"],  # NO

    ["im_semi", "If Quab causes Blaong and Blaong causes the sea level to rise. Does Quab cause the sea level to rise?"],  # YES
    ["im_semi", "If Quab causes Blaong and Aiftan causes the sea level to rise. Does Quab cause the sea level to rise?"],  # NO
    ["im_semi", "If rising temperatures cause Quab and Quab causes the sea level to rise. Do rising temperatures cause the sea level to rise?"],  # YES
    ["im_semi", "If rising temperatures cause Quab and Aiftan causes the sea level to rise. Does rising temperatures cause the sea level to rise?"],  # NO
]

causal_world_questions = [data[1] for data in causal_world_data]


def generate_causal_chains_questions(prefix="", postfix=""):
    question_instances = []

    for question_str in causal_world_questions:
        info = {
            "template": question_str
        }
        question = prefix + question_str + postfix

        question_instances.append({
            "question": question,
            "info": info
        })

    return question_instances


def main():
    question_instances = generate_causal_chains_questions()
    if not dry_run:
        store_query_instances(queries_path, question_instances)

    # chain of thought
    for i in range(1, 5):
        cot_prefix = create_cot_world_prefix(i)
        question_instances = generate_causal_chains_questions(prefix=cot_prefix, postfix=cot_world_postfix)
        if not dry_run:
            store_query_instances(queries_path+f"_cot_{i}", question_instances)

    print("done.")


if __name__ == "__main__":
    main()
