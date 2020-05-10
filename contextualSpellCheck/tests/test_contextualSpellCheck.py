import pytest
import spacy

from contextualSpellCheck.contextualSpellCheck import ContextualSpellCheck

# print(contextualSpellCheck.__name__,contextualSpellCheck.__package__,contextualSpellCheck.__file__,sep="\n")
# This is the class we want to test. So, we need to import it


nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser"])

checker = ContextualSpellCheck()  # instantiate the Person Class


@pytest.mark.parametrize(
    "inputSentence, misspell",
    [
        ("Income was $9.4 million compared to the prior year of $2.7 million.", []),
        ("who is Rajat Goel?", []),
        ("He released this package in year 2020!", []),
    ],
)
def test_no_misspellIdentify(inputSentence, misspell):
    print("Start no spelling mistake test\n")
    doc = nlp(inputSentence)
    assert checker.misspellIdentify(doc) == (misspell, doc)


@pytest.mark.parametrize(
    "inputSentence, misspell",
    [("Income was $9.4 milion compared to the prior year of $2.7 milion.", [4, 13])],
)
def test_type_misspellIdentify(inputSentence, misspell):
    print("Start type correction test for spelling mistake identification\n")
    doc = nlp(inputSentence)
    assert type(checker.misspellIdentify(doc)[0]) == type(misspell)
    assert type(checker.misspellIdentify(doc)[1]) == type(doc)
    assert checker.misspellIdentify(doc)[1] == doc


@pytest.mark.parametrize(
    "inputSentence, misspell",
    [
        ("Income was $9.4 milion compared to the prior year of $2.7 milion.", [4, 13]),
        ("This packge was cretaed in 2020", [1, 3]),
    ],
)
def test_identify_misspellIdentify(inputSentence, misspell):
    print("Start misspell word identifation test\n")
    doc = nlp(inputSentence)
    assert checker.misspellIdentify(doc)[0] == [doc[i] for i in misspell]


@pytest.mark.parametrize(
    "inputSentence, misspell",
    [
        ("Income was $9.4 milion compared to the prior year of $2.7 milion.", 3),
        ("Income was $9.4 milion compared to the prior year of $2.7 milion.", 12),
        ("This packge was cretaed in 2020", 5),
    ],
)
def test_skipNumber_misspellIdentify(inputSentence, misspell):
    print("Start number not in misspell word test\n")
    doc = nlp(inputSentence)
    # Number should not be skipped for misspell
    assert doc[misspell] not in checker.misspellIdentify(doc)[0]


@pytest.mark.parametrize(
    "inputSentence, misspell",
    [
        ("Mr Bond should be skipped", 1),
        ("Amitabh Bachan should not be in mis spell", 0),
        ("Amitabh Bachan shuld not be in mis spell", 1),
    ],
)
def test_skipName_misspellIdentify(inputSentence, misspell):
    print("Start name not in misspell word test\n")
    doc = nlp(inputSentence)
    # Number should not be skipped for misspell
    assert doc[misspell] not in checker.misspellIdentify(doc)[0]


@pytest.mark.parametrize(
    "inputSentence, misspell",
    [
        ("Bond@movies.com should be skipped", 0),
        ("Amitabh.Bachan@bollywood.in should not be in mis spell", 0),
    ],
)
def test_skipEmail_misspellIdentify(inputSentence, misspell):
    print("Start Email not in misspell word test\n")
    doc = nlp(inputSentence)
    assert doc[misspell] not in checker.misspellIdentify(doc)[0]


@pytest.mark.parametrize(
    "inputSentence, misspell",
    [
        ("eng-movies.com should be skipped", 0),
        ("bollywood.in should not be in mis spell", 0),
    ],
)
def test_skipURL_misspellIdentify(inputSentence, misspell):
    print("Start URL not in misspell word test\n")
    doc = nlp(inputSentence)
    assert doc[misspell] not in checker.misspellIdentify(doc)[0]


@pytest.mark.parametrize(
    "inputSentence, misspell",
    [
        ("eng-movies.com shuld be skipped", 0),
        ("bollywood.in shuld not be in mis spell", 0),
    ],
)
def test_type_candidateGenerator(inputSentence, misspell):
    doc = nlp(inputSentence)
    misspell, doc = checker.misspellIdentify(doc)
    assert type(checker.candidateGenerator(doc, misspell)) == dict


@pytest.mark.parametrize(
    "inputSentence, misspell",
    [
        (
            "Income was $9.4 milion compared to the prior year of $2.7 milion.",
            {
                4: [
                    "million",
                    "billion",
                    ",",
                    "trillion",
                    "Million",
                    "%",
                    "##M",
                    "annually",
                    "##B",
                    "USD",
                ],
                13: [
                    "billion",
                    "million",
                    "trillion",
                    "##M",
                    "Million",
                    "##B",
                    "USD",
                    "##b",
                    "millions",
                    "%",
                ],
            },
        ),
        (
            "This packge was introduced in 2020",
            {
                1: [
                    "system",
                    "model",
                    "version",
                    "technology",
                    "program",
                    "standard",
                    "class",
                    "feature",
                    "plan",
                    "service",
                ]
            },
        ),
    ],
)
def test_identify_candidateGenerator(inputSentence, misspell):
    print("Start misspell word identifation test\n")
    doc = nlp(inputSentence)
    (misspellings, doc) = checker.misspellIdentify(doc)
    suggestions = checker.candidateGenerator(doc, misspellings)
    gold_suggestions = {doc[key]: value for key, value in misspell.items()}
    assert suggestions == gold_suggestions


@pytest.mark.parametrize(
    "inputSentence, misspell",
    [
        ("Income was $9.4 milion compared to the prior year of $2.7 milion.", True),
        ("This package was introduced in 2020", False),
    ],
)
def test_extension_candidateGenerator(inputSentence, misspell):
    doc = nlp(inputSentence)
    (misspellings, doc) = checker.misspellIdentify(doc)
    suggestions = checker.candidateGenerator(doc, misspellings)
    assert doc._.performed_spellCheck == misspell


@pytest.mark.parametrize(
    "inputSentence, misspell",
    [
        (
            "Income was $9.4 milion compared to the prior year of $2.7 milion.",
            {
                4: [
                    ("million", 0.59422),
                    ("billion", 0.24349),
                    (",", 0.08809),
                    ("trillion", 0.01835),
                    ("Million", 0.00826),
                    ("%", 0.00672),
                    ("##M", 0.00591),
                    ("annually", 0.0038),
                    ("##B", 0.00205),
                    ("USD", 0.00113),
                ],
                13: [
                    ("billion", 0.65934),
                    ("million", 0.26185),
                    ("trillion", 0.05391),
                    ("##M", 0.0051),
                    ("Million", 0.00425),
                    ("##B", 0.00268),
                    ("USD", 0.00153),
                    ("##b", 0.00077),
                    ("millions", 0.00059),
                    ("%", 0.00041),
                ],
            },
        ),
        (
            "This packge was introduced in 2020",
            {
                1: [
                    ("system", 0.0876),
                    ("model", 0.04924),
                    ("version", 0.04367),
                    ("technology", 0.03086),
                    ("program", 0.01936),
                    ("standard", 0.01607),
                    ("class", 0.01557),
                    ("feature", 0.01527),
                    ("plan", 0.01435),
                    ("service", 0.01351),
                ]
            },
        ),
    ],
)
def test_extension2_candidateGenerator(inputSentence, misspell):
    doc = nlp(inputSentence)
    (misspellings, doc) = checker.misspellIdentify(doc)
    suggestions = checker.candidateGenerator(doc, misspellings)
    assert doc._.score_spellCheck == {
        doc[key]: value for key, value in misspell.items()
    }


@pytest.mark.parametrize(
    "inputSentence, misspell",
    [
        (
            "Income was $9.4 milion compared to the prior year of $2.7 milion.",
            {4: "million", 13: "million"},
        ),
        ("This package was introduced in 2020", {}),
    ],
)
def test_ranking_candidateRanking(inputSentence, misspell):
    doc = nlp(inputSentence)
    (misspellings, doc) = checker.misspellIdentify(doc)
    suggestions = checker.candidateGenerator(doc, misspellings)
    selectedWord = checker.candidateRanking(suggestions)
    assert selectedWord == {doc[key]: value for key, value in misspell.items()}


def test_compatible_spacyPipeline():
    nlp.add_pipe(checker)
    assert "contextual spellchecker" in nlp.pipe_names

    nlp.remove_pipe("contextual spellchecker")
    assert "contextual spellchecker" not in nlp.pipe_names


def test_doc_extensions():
    nlp.add_pipe(checker)
    doc = nlp(u"Income was $9.4 milion compared to the prior year of $2.7 milion.")

    gold_suggestion = {
        doc[4]: [
            "million",
            "billion",
            ",",
            "trillion",
            "Million",
            "%",
            "##M",
            "annually",
            "##B",
            "USD",
        ],
        doc[13]: [
            "billion",
            "million",
            "trillion",
            "##M",
            "Million",
            "##B",
            "USD",
            "##b",
            "millions",
            "%",
        ],
    }
    gold_outcome = "Income was $9.4 million compared to the prior year of $2.7 million."
    gold_score = {
        doc[4]: [
            ("million", 0.59422),
            ("billion", 0.24349),
            (",", 0.08809),
            ("trillion", 0.01835),
            ("Million", 0.00826),
            ("%", 0.00672),
            ("##M", 0.00591),
            ("annually", 0.0038),
            ("##B", 0.00205),
            ("USD", 0.00113),
        ],
        doc[13]: [
            ("billion", 0.65934),
            ("million", 0.26185),
            ("trillion", 0.05391),
            ("##M", 0.0051),
            ("Million", 0.00425),
            ("##B", 0.00268),
            ("USD", 0.00153),
            ("##b", 0.00077),
            ("millions", 0.00059),
            ("%", 0.00041),
        ],
    }
    assert doc._.contextual_spellCheck == True
    assert doc._.performed_spellCheck == True
    assert doc._.suggestions_spellCheck == gold_suggestion
    assert doc._.outcome_spellCheck == gold_outcome
    assert doc._.score_spellCheck == gold_score
    nlp.remove_pipe("contextual spellchecker")


def test_span_extensions():
    nlp.add_pipe(checker)
    doc = nlp("Income was $9.4 milion compared to the prior year of $2.7 milion.")

    gold_score = [
        {doc[2]: []},
        {doc[3]: []},
        {
            doc[4]: [
                ("million", 0.59422),
                ("billion", 0.24349),
                (",", 0.08809),
                ("trillion", 0.01835),
                ("Million", 0.00826),
                ("%", 0.00672),
                ("##M", 0.00591),
                ("annually", 0.0038),
                ("##B", 0.00205),
                ("USD", 0.00113),
            ]
        },
        {doc[5]: []},
    ]

    assert doc[2:6]._.get_has_spellCheck == True
    assert doc[2:6]._.score_spellCheck == gold_score
    nlp.remove_pipe("contextual spellchecker")


def test_token_extension():
    if "contextual spellchecker" not in nlp.pipe_names:
        nlp.add_pipe(checker)
    doc = nlp("Income was $9.4 milion compared to the prior year of $2.7 milion.")

    gold_suggestions = [
        "million",
        "billion",
        ",",
        "trillion",
        "Million",
        "%",
        "##M",
        "annually",
        "##B",
        "USD",
    ]
    gold_score = [
        ("million", 0.59422),
        ("billion", 0.24349),
        (",", 0.08809),
        ("trillion", 0.01835),
        ("Million", 0.00826),
        ("%", 0.00672),
        ("##M", 0.00591),
        ("annually", 0.0038),
        ("##B", 0.00205),
        ("USD", 0.00113),
    ]

    assert doc[4]._.get_require_spellCheck == True
    assert doc[4]._.get_suggestion_spellCheck == gold_suggestions
    assert doc[4]._.score_spellCheck == gold_score
    nlp.remove_pipe("contextual spellchecker")
