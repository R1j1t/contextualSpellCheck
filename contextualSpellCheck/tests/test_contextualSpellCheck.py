import pytest
import spacy

from contextualSpellCheck.contextualSpellCheck import ContextualSpellCheck

# print(contextualSpellCheck.__name__,contextualSpellCheck.__package__,contextualSpellCheck.__file__,sep="\n")
# This is the class we want to test. So, we need to import it


nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser"])

checker = ContextualSpellCheck()  # instantiate the Person Class
user_id = []  # variable that stores obtained user_id
user_name = []  # variable that stores person name


@pytest.mark.parametrize(
    "inputSentence, misspell",
    [
        ("Income was $9.4 million compared to the prior year of $2.7 million.", []),
        ("who is Rajat Goel?", []),
        ("He released this package in year 2020!", []),
    ],
)
def test_no_misspellIdentify(inputSentence, misspell):
    print("Start misspellIdentify test\n")
    doc = nlp(inputSentence)
    assert checker.misspellIdentify(doc) == (misspell, doc)


@pytest.mark.parametrize(
    "inputSentence, misspell",
    [("Income was $9.4 million compared to the prior year of $2.7 million.", [])],
)
def test_type_misspellIdentify(inputSentence, misspell):
    print("Start misspellIdentify test\n")
    doc = nlp(inputSentence)
    assert type(checker.misspellIdentify(doc)[0]) == type(misspell)
    assert type(checker.misspellIdentify(doc)[1]) == type(doc)
    assert checker.misspellIdentify(doc)[1] == doc
