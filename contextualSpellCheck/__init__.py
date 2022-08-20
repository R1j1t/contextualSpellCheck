"""
Provides a high level function (`add_to_pipe`)
to access/initialize contextualSpellCheck
"""

# pylint: disable=invalid-name
import spacy
from .contextualSpellCheck import ContextualSpellCheck

__all__ = ["ContextualSpellCheck", "add_to_pipe"]


def add_to_pipe(nlp: spacy.language.Language, **kwargs):
    """Function to add contextualSpellCheck to spacy pipeline

    Args:
        nlp (spacy.language.Language): spacy pipeline
    """
    nlp.add_pipe("contextual spellchecker", **kwargs)
