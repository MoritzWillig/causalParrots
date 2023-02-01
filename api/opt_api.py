from typing import Optional


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from rtpt import RTPT

from causalFM.api.languageModelInterface import LanguageModelInterface

opt_model = "facebook/opt-30b" # "facebook/opt-30b"

opt_device_map = "auto"  # distribute parameter across all available gpus (and cpu if needed)
#device_map = None  # single gpu


def startup_opt(key):
    rtpt = RTPT(name_initials='XXX', experiment_name='', max_iterations=100)
    rtpt.start()

    model = AutoModelForCausalLM.from_pretrained(opt_model, torch_dtype=torch.float16, device_map=opt_device_map).cuda()

    # the fast tokenizer currently does not work correctly
    tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)

    return model, tokenizer


def query_opt(context, query_text, dry_run=False):
    model, tokenizer = context

    if dry_run:
        return None
    input_ids = tokenizer(query_text, return_tensors="pt").input_ids.cuda()
    generated_ids = model.generate(input_ids, num_return_sequences=1, max_length=70)

    results = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return results[0][len(query_text):]


class OptLM(LanguageModelInterface):

    def __init__(self, key: str = None, limit: Optional[int] = 5, dry_run: bool = False):
        super().__init__(key, limit, dry_run)
        #self.context = startup_opt(key)
        raise NotImplementedError()

    def do_query(self, query_text: str) -> str:
        #return query_opt(self.context, query_text)
        raise NotImplementedError()

    def do_get_embedding(self, query_text: str) -> list:
        raise NotImplementedError()
