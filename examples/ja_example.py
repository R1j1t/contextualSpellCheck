import spacy
import contextualSpellCheck

nlp = spacy.load("ja_core_news_sm")

nlp.add_pipe(
    "contextual spellchecker",
    config={
        "model_name": "cl-tohoku/bert-base-japanese-whole-word-masking",
        "max_edit_dist": 2,
    },
)

doc = nlp("しかし大勢においては、ここような事故はウィキペディアの拡大には影響を及ぼしていない。")
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
