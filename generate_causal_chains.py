from causalFM.query_helpers import questions, AttrDict, instantiate_questions, store_query_instances

dry_run = False
queries_path = "./queries/causal_chains"

cot_cond_texts = [
"""Q: If X causes Y and Y causes Z. Does X cause Z?
A: Because X causes Y and Y causes Z, X causes Z. The answer is yes.""",

"""Q: If Z causes Y and Y causes X. Does Z cause X?
A: Because Z causes Y and Y causes X, Z causes X. The answer is yes.""",

"""Q: If X causes Y and Y causes Z. Does Y cause X?
A: Because Y does not cause X directly and Y causes Z which does not cause X, Y does not cause X. The answer is no.""",

"""Q: If Y causes Z and X causes Y. Does X cause Z?
A: Because X causes Y and Y causes Z, X causes Z. The answer is yes.""",

"""Q: If Y causes Z, W causes X, X causes Y and V causes W. Does M cause Z?
A: Because M does not appear in any of the clauses, the answer is no.""",

"""Q: If Y causes Z, W causes X, X causes Y and V causes W. Does V cause Z?
A: Because V causes W, W causes X, X causes Y and Y causes Z, the answer is yes.""",

"""Q: If V causes W, W causes X, X causes Y and Y causes Z. Does X cause W?
A: Because X only causes Y and Y only causes Z, there is no directed path from X to W. The answer is no.""",

"""Q: If Y causes Z, W causes X and V causes W. Does V cause Z?
A: V causes W and W causes X, but neither V, W nor X cause Z. The answer is no.""",
]
cot_chain_postfix = "\nA: "


def create_cot_chain_prefix(n):
    cot_prefix = "\n\n".join(cot_cond_texts[:n]) + "\n\nQ:"
    return cot_prefix


causal_chains_data = [
    # reasoning on A->B->C chain
    ["chain", "If A causes B and B causes C. Does A cause C?"],  # A->B->C. A->C? # 3  YES
    ["sub", "If A causes B and B causes C. Does A cause B?"],  # A->B->C. A->B?        YES
    ["sub", "If A causes B and B causes C. Does B cause C?"],  # A->B->C. B->C?        YES
    ["sub", "If A causes B and B causes C. Does A cause A?"],  # A->B->C. A->A?        NO
    ["rand", "If A causes B and B causes C. Does B cause A?"],  # A->B->C. B->A?       NO
    ["rand", "If A causes B and B causes C. Does C cause A?"],  # A->B->C. C->A?       NO

    # extending chain
    ["chain", "If A causes B, B causes C and C causes D. Does A cause D?"],  # 4                       YES
    ["chain", "If A causes B, B causes C, C causes D and D causes E. Does A cause E?"],  # 5           YES
    ["chain", "If A causes B, B causes C, C causes D, D causes E, E causes F. Does A cause F?"],  # 6  YES
    ["sub", "If A causes B, B causes C, C causes D, D causes E, E causes F. Does B cause E?"],  #      YES
    ["rand", "If A causes B, B causes C, C causes D, D causes E, E causes F. Does E cause B?"],  #     NO

    # changing clause order
    ["rand", "If B causes C and A causes B. Does A cause C?"],  # B->C, A->B. A->C?    YES
    ["rand", "If B causes C and A causes B. Does C cause A?"],  # B->C, A->B. C->A?    NO

    # changing variable names
    ["rand", "If G causes Q and Q causes S. Does G cause S?"],  # G->Q->S. G->S?       YES

    # changing clause order and rename
    ["rand", "If Q causes S and G causes Q . Does G cause S?"],  # Q->S, G->Q. G->S?   YES
    
    # extending chain continued
    ["chain", "If A causes B. Does A cause B?"],  # 2                                                                                                   YES
    ["chain", "If A causes B, B causes C, C causes D, D causes E, E causes F, F causes G. Does A cause G?"],  # 7                                       YES
    ["chain", "If A causes B, B causes C, C causes D, D causes E, E causes F, F causes G, G causes H. Does A cause H?"],  # 8                           YES
    ["chain", "If A causes B, B causes C, C causes D, D causes E, E causes F, F causes G, G causes H, H causes I. Does A cause I?"],  # 9               YES
    ["chain", "If A causes B, B causes C, C causes D, D causes E, E causes F, F causes G, G causes H, H causes I, I causes J. Does A cause J?"]  # 10   YES
]

causal_chains_questions = [data[1] for data in causal_chains_data]


def generate_causal_chains_questions(prefix="", postfix=""):
    question_instances = []

    for question_str in causal_chains_questions:
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
    #question_instances = generate_causal_chains_questions()
    #if not dry_run:
    #    store_query_instances(queries_path, question_instances)

    # chain of thought
    for i in range(1, 9):
        cot_prefix = create_cot_chain_prefix(i)
        question_instances = generate_causal_chains_questions(prefix=cot_prefix, postfix=cot_chain_postfix)
        if not dry_run:
            store_query_instances(queries_path+f"_cot_{i}", question_instances)

    print("done.")


if __name__ == "__main__":
    main()
