from causalFM.query_helpers import questions, AttrDict, instantiate_questions, store_query_instances

dry_run = False
queries_path = "./queries/causal_chains"

causal_chains_data = [
    # reasoning on A->B->C chain
    ["chain", "If A causes B and B causes C. Does A cause C?"],  # A->B->C. A->C? # 3
    ["sub", "If A causes B and B causes C. Does A cause B?"],  # A->B->C. A->B?
    ["sub", "If A causes B and B causes C. Does B cause C?"],  # A->B->C. B->C?
    ["rand", "If A causes B and B causes C. Does A cause A?"],  # A->B->C. A->A?
    ["rand", "If A causes B and B causes C. Does B cause A?"],  # A->B->C. B->A?
    ["rand", "If A causes B and B causes C. Does C cause A?"],  # A->B->C. C->A?

    # extending chain
    ["chain", "If A causes B, B causes C and C causes D. Does A cause D?"],  # 4
    ["chain", "If A causes B, B causes C, C causes D and D causes E. Does A cause E?"],  # 5
    ["chain", "If A causes B, B causes C, C causes D, D causes E, E causes F. Does A cause F?"],  # 6
    ["sub", "If A causes B, B causes C, C causes D, D causes E, E causes F. Does B cause E?"],
    ["sub", "If A causes B, B causes C, C causes D, D causes E, E causes F. Does E cause B?"],

    # changing clause order
    ["rand", "If B causes C and A causes B. Does A cause C?"],  # B->C, A->B. A->C?
    ["rand", "If B causes C and A causes B. Does C cause A?"],  # B->C, A->B. C->A?

    # changing variable names
    ["rand", "If G causes Q and Q causes S. Does G cause S?"],  # G->Q->S. G->S?

    # changing clause order and rename
    ["rand", "If Q causes S and G causes Q . Does G cause S?"],  # Q->S, G->Q. G->S?
    
    # extending chain continued
    ["chain", "If A causes B. Does A cause B?"],  # 2
    ["chain", "If A causes B, B causes C, C causes D, D causes E, E causes F, F causes G. Does A cause G?"],  # 7
    ["chain", "If A causes B, B causes C, C causes D, D causes E, E causes F, F causes G, G causes H. Does A cause H?"],  # 8
    ["chain", "If A causes B, B causes C, C causes D, D causes E, E causes F, F causes G, G causes H, H causes I. Does A cause I?"],  # 9
    ["chain", "If A causes B, B causes C, C causes D, D causes E, E causes F, F causes G, G causes H, H causes I, I causes J. Does A cause J?"]  # 10
]

causal_chains_questions = [data[1] for data in causal_chains_data]


def generate_causal_chains_questions():
    question_instances = []

    for question_str in causal_chains_questions:
        info = {
            "template": question_str
        }
        question = question_str

        question_instances.append({
            "question": question,
            "info": info
        })

    return question_instances


def main():
    question_instances = generate_causal_chains_questions()
    if not dry_run:
        store_query_instances(queries_path, question_instances)
    print("done.")


if __name__ == "__main__":
    main()
