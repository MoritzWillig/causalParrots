import os
import pickle
from pathlib import Path
from typing import Union
import re


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @staticmethod
    def make(simple_dict):
        return AttrDict(**simple_dict)


questions = [
    AttrDict.make({
        "name": "related",
        "pattern": "Are {%1@the} and {%2@bthe} causally related?",
        "direction": "symmetric",
        "modifiers": ["QA", "YesNo", "captialize"],
    }),
    AttrDict.make({
        "name": "connection",
        "pattern": "Is there a causal connection between {%1@the} and {%2@the}?",
        "direction": "symmetric",
        "modifiers": ["QA", "YesNo", "captialize"],
    }),
    AttrDict.make({
        "name": "cause",
        "pattern": "{%1:singular?Does:Do} {%1@the} cause {%2@the}?",
        "direction": "directed",
        "modifiers": ["QA", "YesNo", "captialize"],
    }),
    AttrDict.make({
        "name": "influence",
        "pattern": "{%1:singular?Does:Do} {%1@the} influence {%2@the}?",
        "direction": "directed",
        "modifiers": ["QA", "YesNo", "captialize"],
    }),
    AttrDict.make({
        "name": "causality",
        "pattern": "Is there causality between {%1@the} and {%2@the}?",
        "direction": "symmetric",
        "modifiers": ["QA", "YesNo", "captialize"],
    })
]
question_templates = questions #FIXME remove 'questions' and only use 'question_templates'


cases_statement_template = AttrDict.make({
        "name": "cause",
        "pattern": "{%1@the} {%1:singular?causes:cause} {%2@the}.",
        "direction": "directed",
        "modifiers": ["captialize"],
    })

statement_templates = [
    AttrDict.make({
        "name": "related",
        "pattern": "{%1@the} and {%2@bthe} are causally related.",
        "direction": "symmetric",
        "modifiers": ["captialize"],
    }),
    AttrDict.make({
        "name": "connection",
        "pattern": "There is a causal connection between {%1@the} and {%2@the}.",
        "direction": "symmetric",
        "modifiers": ["captialize"],
    }),
    cases_statement_template,
    AttrDict.make({
        "name": "influence",
        "pattern": "{%1@the} {%1:singular?influences:influence} {%2@the}.",
        "direction": "directed",
        "modifiers": ["captialize"],
    }),
    AttrDict.make({
        "name": "causality",
        "pattern": "There is causality between {%1@the} and {%2@the}.",
        "direction": "symmetric",
        "modifiers": ["captialize"],
    })
]


def get_var_str(var, hintThe=False):
    theFlag = False
    if hintThe and "optionalThe" in var and var.optionalThe:
        s = "the "
        theFlag = True
    else:
        s = ""
    s += var.expression
    return s, theFlag


def instantiate_question(question_template, parts, temp_vars, prevent_modfiers=None):
    assert len(temp_vars) == 2

    question = []
    theFlag = False
    info = {
        "template": question_template["pattern"],
        "names": [temp_vars[0].name, temp_vars[1].name],
        "alt_name": [temp_vars[0].alt_name if "alt_name" in temp_vars[0] else None, temp_vars[1].alt_name if "alt_name" in temp_vars[1] else None],
        "exprs": [temp_vars[0].expression, temp_vars[1].expression],
    }

    for part in parts:
        if isinstance(part, str):
            question.append(part)
        else:
            if part[0] == 'simple':
                idx = int(part[1]) - 1
                var = temp_vars[idx]

                part, theFlag = get_var_str(var)
                question.append(part)
            elif part[0] == 'cond':
                part = part[1]

                cond_split = part.split(":", 1)
                if len(cond_split) == 2:
                    # var condition
                    name, rest = cond_split

                    cond, rest = rest.split("?")
                    ifTrue, ifFalse = rest.split(":")

                    idx = int(name) - 1
                    var = temp_vars[idx]
                    if var[cond]:
                        question.append(ifTrue)
                    else:
                        question.append(ifFalse)
                else:
                    # var with hint
                    name = cond_split[0]

                    name_split = name.split("@", 1)
                    if len(name_split) == 1:
                        raise RuntimeError(f"malformed expression: {part}")
                    name, hints = name_split
                    hints = hints.split(",")

                    idx = int(name) - 1
                    var = temp_vars[idx]
                    # "the"-flag: hints at using "the" before var.
                    # "bthe"-flag: use "the" before var, if the previous variable didn't.
                    part, theFlag = get_var_str(var, hintThe="the" in hints or (not theFlag and "bthe" in hints))
                    question.append(part)

    question = "".join(question)

    modifiers = set(question_template["modifiers"])
    if prevent_modfiers is not None:
        modifiers -= set(prevent_modfiers)

    if "captialize":
        question = question.capitalize()
    if "YesNo" in modifiers:
        question = f"{question} Answer Yes or No."
    if "QA" in modifiers:
        question = f"Q: {question}\nA: "

    print("[instance]", question)
    question_instance = {
        "question": question,
        "info": info
    }
    return question_instance


def template_to_parts(template):
    parts = []
    s_idx = 0
    while True:
        idx = template.find("%", s_idx)
        if idx == -1:
            parts.append(template[s_idx:])
            break
        idx += 1
        if (idx != 1) and (template[idx - 2] != "{"):
            parts.append(template[s_idx:idx - 1])
            sub = ""
            while (idx < len(template)) and (template[idx].isdigit()):
                sub += template[idx]
                idx += 1
            parts.append(('simple', int(sub)))
            s_idx = idx
        else:
            parts.append(template[s_idx:idx - 2])
            e_idx = template.find("}", idx)
            parts.append(('cond', template[idx:e_idx]))
            s_idx = e_idx + 1
    return parts


def instantiate_question_pairs(questions_templates, variable_pairs, prevent_modfiers=None,
                          constrain_to_var=None, generate_inverse=False):
    """
    :param questions_templates:
    :param variables: list of variable pairs
    :param prevent_modfiers: List of str. Remove the given modifiers from the question modifiers
    :param constrain_to_var: only generate instances that contain the variable in any of its parameters
    :return:
    """
    # match %1 or {%1:Q?T:F}
    question_instances = []
    for question_template in questions_templates:
        template = question_template["pattern"]
        parts = template_to_parts(template)
        print(parts)

        temp_vars = ["", ""]
        for v0, v1 in variable_pairs:
            temp_vars[0] = v0
            temp_vars[1] = v1

            if constrain_to_var is not None:
                # skip all questions that do not contain the variable
                if v0.name != constrain_to_var and v1.name != constrain_to_var:
                    continue

            question_instances.append(instantiate_question(question_template, parts, temp_vars, prevent_modfiers=prevent_modfiers))
            if generate_inverse:
                temp_vars[0] = v1
                temp_vars[1] = v0
                question_instances.append(
                    instantiate_question(question_template, parts, temp_vars, prevent_modfiers=prevent_modfiers))
    return question_instances


def instantiate_questions(questions_templates, variables, prevent_modfiers=None, generate_self_references=False,
                          constrain_to_var=None):
    """
    :param questions_templates:
    :param variables:
    :param prevent_modfiers: List of str. Remove the given modifiers from the question modifiers
    :param generate_self_references: If true, generates questions that query self references (A->A connections)
    :param constrain_to_var: only generate instances that contain the variable in any of its parameters
    :return:
    """
    #match %1 or {%1:Q?T:F}
    question_instances = []
    for question_template in questions_templates:
        template = question_template["pattern"]
        parts = template_to_parts(template)
        print(parts)

        temp_vars = ["", ""]
        for v0 in variables:
            temp_vars[0] = v0
            for v1 in variables:
                temp_vars[1] = v1

                if not generate_self_references and v0 == v1:
                    continue

                if constrain_to_var is not None:
                    # skip all questions that do not contain the variable
                    if v0.name != constrain_to_var and v1.name != constrain_to_var:
                        continue

                question_instances.append(instantiate_question(question_template, parts, temp_vars, prevent_modfiers=prevent_modfiers))
    return question_instances


def store_query_instances(path: Union[str, Path], question_instances):
    if not isinstance(path, Path):
        path = Path(path)
        path.parent.mkdir(exist_ok=True)

    with (path.parent / (path.name + "_full.pkl")).open("wb+") as f:
        pickle.dump(question_instances, f)

    with (path.parent / (path.name + "_questions.txt")).open("w+") as f:
        for instance in question_instances:
            query = instance["question"]
            query = query.replace("\n", "\\n")
            query = re.sub(r"\ +", " ", query)
            f.write(query + "\n")


def load_query_instances(path):
    if isinstance(path, str):
        path = Path(path)
    with path.open("r") as f:
        lines = f.readlines()
    lines = [line.strip().replace("\\n", "\n") for line in lines]
    return lines
