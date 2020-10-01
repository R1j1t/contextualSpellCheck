from contextualSpellCheck.contextualSpellCheck import ContextualSpellCheck

__all__ = ["ContextualSpellCheck", "add_to_pipe"]


def add_to_pipe(nlp):
    checker = ContextualSpellCheck()
    nlp.add_pipe(checker)
    return nlp
