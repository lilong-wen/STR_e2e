# query based weakly supervised sence text retrieval
# aka Quester
from .quester import build


def build_model(args):
    return build(args)
