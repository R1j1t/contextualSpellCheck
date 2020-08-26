from .contextualSpellCheck import ContextualSpellCheck
import spacy

__all__ = ["ContextualSpellCheck", "add_to_pipe"]


def add_to_pipe(nlp, **kwargs):
    checker = ContextualSpellCheck(**kwargs)
    nlp.add_pipe(checker)
    return nlp
