import spacy
import contextualSpellCheck

nlp = spacy.load("en_core_web_sm")

nlp.add_pipe("contextual spellchecker", config={"max_edit_dist": 100})

doc = nlp("Income was $9.4 milion compared to the prior year of $2.7 milion.")
print(doc._.performed_spellCheck)
print(doc._.outcome_spellCheck)

# Doc Extention
print(doc._.contextual_spellCheck)

print(doc._.performed_spellCheck)

print(doc._.suggestions_spellCheck)

print(doc._.outcome_spellCheck)

print(doc._.score_spellCheck)

# Token Extention
print(doc[4]._.get_require_spellCheck)

print(doc[4]._.get_suggestion_spellCheck)

print(doc[4]._.score_spellCheck)

# Span Extention
print(doc[2:6]._.get_has_spellCheck)

print(doc[2:6]._.score_spellCheck)
