from causalFM.api.aleph_alpha_api import AlphaAlphaLM
from causalFM.api.openai_api import OpenAILM, OpenAILMAda
#from causalFM.api.opt_api import OptLM
#FIXME
OptLM = None

lm_apis = {
    "openai": {
        "model": OpenAILM
    },
    "openai_textEmbAda002": {
        "model": OpenAILMAda
    },
    "aleph_alpha": {
        "model": AlphaAlphaLM
    },
    "opt": {
        "model": OptLM
    }
}
