from causalFM.query_helpers import questions, AttrDict, instantiate_questions, store_query_instances

dry_run = False
queries_path = "./queries/intuitive_physics"

physics_data = [
    # rolling & falling
    ["rolling", "There is a tilted board above a bucket. Where does a ball end up if it is placed on the board?"],
    ["rolling", "There is a tilted board above a bucket. Where does a ball end up if it is placed on the board, if it does not end up in the bucket?"],
    ["rolling", "There is a board above a bucket. Where does a ball end up if it is placed on the board?"],
    ["rolling", "There is a leveled board above a bucket. Where does a ball end up if it is placed on the board?"],

    # inference
    ["rolling", "A ball is placed on a table and rolls off. What does this tell us about the table?"],
    ["rolling", "A ball is placed on a surface and rolls off. What does this tell us about the surface?"],
    ["rolling", "A ball is placed on a table and rolls to the center. What does this tell us about the table?"],
    ["rolling", "A ball is placed on a surface and rolls to the center. What does this tell us about the surface?"],

    # object support generic
    ["support", "A block is placed on one support. What happens if the support is removed?"],
    ["support", "A block is placed on two supports. What happens if one of the supports is removed?"],
    ["support", "A block is placed on three supports. What happens if one of the supports is removed?"],
    ["support", "A block is placed on three supports. What happens if the left support is removed?"],
    ["support", "A block is placed on three supports. What happens if the right support is removed?"],
    ["support", "A block is placed on three supports. What happens if the middle support is removed?"],

    # object support
    ["support", "A vase is supported by a table. What happens if the table is removed?"],
    ["support", "An vase is supported by a table. What happens if the table disappears?"],

    # collisions
    ["collisions", "A heavy and a light ball are heading towards each other. Which ball will bounce away?"],  # The light ball will bounce away.
    ["collisions", "Two equal weight balls are heading towards each other. Which ball will bounce away?"],  # equal weight
    ["collisions", "Two balls are heading towards each other with the same speed. Which ball will bounce away?"],  # equal velocity
    ["collisions", "Two equal weight balls are heading towards each other with the same speed. Which ball will bounce away?"],  # equal weight and velocity

    # seesaw / levers
    ["seesaw", "A heavy and a light object are placed on a seesaw. Which object will move up?"],  # note that the setup is technically not well defined
    ["seesaw", "A heavy and a light object are placed on a seesaw. Which object will move down?"],  # reverse of before
    ["seesaw", "Two equal weight object are placed on a seesaw. Which object will move up?"],
    ["seesaw", "A heavy and a light object are placed on the same side of a seesaw. Which object will move down?"],  # 'trick' question

    # weight comparisons
    ["weights", "What is heavier: A hand full of metal or a hand full of feathers?"],
    ["weights", "What is heavier: A kilogram of metal or a kilogram of feathers?"],  # ... "A kilo of metal is heavier than a kilo of feathers."
    ["weights", "What is heavier: A kilogram of metal or a kilogram of rock?"],  # about equal weight??
    ["weights", "What is heavier: A kilogram of metal or a kilogram of lead?"],  # higher specific weight for lead
    ["weights", '"A kilogram of metal is heavier than a kilogram of feathers" is what most people say, but in reality'],  # give them a chance

    # make-shift levers
    ["tools", "A wooden beam is placed over a stone. A small plastic bottle is placed on one side. What will happen if a person jumps onto the other side of the beam?"],

    # mary move stone
    ["tools", "Mary can not move a heavy stone by herself. However, she brought a small object and a metal rod with her. Is Mary able to move the stone?"],
    ["tools", "Mary can not move a heavy stone by herself. However, she brought a small object and a metal rod with her. How can Mary move the stone?"], # introduce intention

    # mary lift stone
    ["tools", "Mary can not lift a heavy stone by herself. However, she brought a small object and a metal rod with her. Is Mary able to lift the stone?"],
    ["tools", "Mary can not lift a heavy stone by herself. However, she brought a small object and metal rod with her. How can Mary lift the stone?"],

    # mary brought an unhelpful object
    ["tools", "Mary can not lift a heavy stone by herself. However, she brought a calculator with her. Is Mary able to lift the stone?"],
    ["tools", "Mary can not lift a heavy stone by herself. However, she brought a calculator with her. How can Mary lift the stone?"],
]

physics_questions = [data[1] for data in physics_data]


def generate_physic_questions():
    question_instances = []

    for question_str in physics_questions:
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
    question_instances = generate_physic_questions()
    if not dry_run:
        store_query_instances(queries_path, question_instances)
    print("done.")


if __name__ == "__main__":
    main()
