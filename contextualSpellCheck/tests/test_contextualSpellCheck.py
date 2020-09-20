import pytest
import spacy
from pytest import approx
import warnings, os

from ..contextualSpellCheck import ContextualSpellCheck

# print(contextualSpellCheck.__name__,contextualSpellCheck.__package__,contextualSpellCheck.__file__,sep="\n")
# This is the class we want to test. So, we need to import it


nlp = spacy.load("en_core_web_sm")

checker = ContextualSpellCheck()  # instantiate the Person Class


@pytest.mark.parametrize(
    "inputSentence, misspell",
    [
        (
            "Income was $9.4 million compared to the prior year of $2.7 million.",
            [],
        ),
        ("who is Rajat Goel?", []),
        ("He released this package in year 2020!", []),
    ],
)
def test_no_misspellIdentify(inputSentence, misspell):
    print("Start no spelling mistake test\n")
    doc = nlp(inputSentence)
    assert checker.misspell_identify(doc) == (misspell, doc)


@pytest.mark.parametrize(
    "inputSentence, misspell",
    [
        (
            "Income was $9.4 milion compared to the prior year of $2.7 milion.",
            [4, 13],
        )
    ],
)
def test_type_misspellIdentify(inputSentence, misspell):
    print("Start type correction test for spelling mistake identification\n")
    doc = nlp(inputSentence)
    assert type(checker.misspell_identify(doc)[0]) == type(misspell)
    assert type(checker.misspell_identify(doc)[1]) == type(doc)
    assert checker.misspell_identify(doc)[1] == doc


@pytest.mark.parametrize(
    "inputSentence, misspell",
    [
        (
            "Income was $9.4 milion compared to the prior year of $2.7 milion.",
            [4, 13],
        ),
        ("This packge was cretaed in 2020", [1, 3]),
    ],
)
def test_identify_misspellIdentify(inputSentence, misspell):
    print("Start misspell word identifation test\n")
    doc = nlp(inputSentence)
    checkerReturn = checker.misspell_identify(doc)[0]
    assert type(checkerReturn) == list
    ## Changed the approach after v0.1.0
    assert [tok.text_with_ws for tok in checkerReturn] == [
        doc[i].text_with_ws for i in misspell
    ]
    assert [tok.i for tok in checkerReturn] == [doc[i].i for i in misspell]


@pytest.mark.parametrize(
    "inputSentence, misspell",
    [
        (
            "Income was $9.4 milion compared to the prior year of $2.7 milion.",
            3,
        ),
        (
            "Income was $9.4 milion compared to the prior year of $2.7 milion.",
            12,
        ),
        ("This packge was cretaed in 2020", 5),
    ],
)
def test_skipNumber_misspellIdentify(inputSentence, misspell):
    print("Start number not in misspell word test\n")
    doc = nlp(inputSentence)
    # Number should not be skipped for misspell
    assert doc[misspell] not in checker.misspell_identify(doc)[0]


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
    assert doc[misspell] not in checker.misspell_identify(doc)[0]


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
    assert doc[misspell] not in checker.misspell_identify(doc)[0]


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
    assert doc[misspell] not in checker.misspell_identify(doc)[0]


@pytest.mark.parametrize(
    "inputSentence, misspell",
    [
        ("eng-movies.com shuld be skipped", 0),
        ("bollywood.in shuld not be in mis spell", 0),
    ],
)
def test_type_candidateGenerator(inputSentence, misspell):
    doc = nlp(inputSentence)
    misspell, doc = checker.misspell_identify(doc)
    assert type(checker.candidate_generator(doc, misspell)) == tuple
    assert type(checker.candidate_generator(doc, misspell)[0]) == type(doc)
    assert type(checker.candidate_generator(doc, misspell)[1]) == dict


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
    (misspellings, doc) = checker.misspell_identify(doc)
    doc, suggestions = checker.candidate_generator(doc, misspellings)
    ## changed after v1.0 because of deepCopy creatng issue with ==
    # gold_suggestions = {doc[key]: value for key, value in misspell.items()}
    assert [tok.i for tok in suggestions] == [key for key in misspell.keys()]
    assert [suggString for suggString in suggestions.values()] == [
        suggString for suggString in misspell.values()
    ]

    # assert suggestions == gold_suggestions


@pytest.mark.parametrize(
    "inputSentence, misspell",
    [
        (
            "Income was $9.4 milion compared to the prior year of $2.7 milion.",
            True,
        ),
        ("This package was introduced in 2020", False),
    ],
)
def test_extension_candidateGenerator(inputSentence, misspell):
    doc = nlp(inputSentence)
    (misspellings, doc) = checker.misspell_identify(doc)
    suggestions = checker.candidate_generator(doc, misspellings)
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
    (misspellings, doc) = checker.misspell_identify(doc)
    doc, suggestions = checker.candidate_generator(doc, misspellings)

    # changes after v0.1.0
    assert [tokIndex.i for tokIndex in doc._.score_spellCheck.keys()] == [
        tokIndex for tokIndex in misspell.keys()
    ]
    assert [
        word_score[0]
        for value in doc._.score_spellCheck.values()
        for word_score in value
    ] == [word_score[0] for value in misspell.values() for word_score in value]
    assert [
        word_score[1]
        for value in doc._.score_spellCheck.values()
        for word_score in value
    ] == approx(
        [word_score[1] for value in misspell.values() for word_score in value],
        rel=1e-4,
        abs=1e-4,
    )


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
    (misspellings, doc) = checker.misspell_identify(doc)
    doc, suggestions = checker.candidate_generator(doc, misspellings)
    selectedWord = checker.candidate_ranking(doc, suggestions)
    ## changes made after v0.1
    # assert selectedWord == {doc[key]: value for key, value in misspell.items()}
    assert [tok.i for tok in selectedWord.keys()] == [
        tok for tok in misspell.keys()
    ]
    assert [tokString for tokString in selectedWord.values()] == [
        tok for tok in misspell.values()
    ]


def test_compatible_spacyPipeline():
    nlp.add_pipe(checker)
    assert "contextual spellchecker" in nlp.pipe_names

    nlp.remove_pipe("contextual spellchecker")
    assert "contextual spellchecker" not in nlp.pipe_names


def test_doc_extensions():
    nlp.add_pipe(checker)
    doc = nlp(
        "Income was $9.4 milion compared to the prior year of $2.7 milion."
    )

    gold_suggestion = {
        doc[4]: "million",
        doc[13]: "million",
    }
    gold_outcome = (
        "Income was $9.4 million compared to the prior year of $2.7 million."
    )
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
    # updated after v0.1
    assert [tok.i for tok in doc._.suggestions_spellCheck.keys()] == [
        tok.i for tok in gold_suggestion.keys()
    ]
    assert [
        tokString for tokString in doc._.suggestions_spellCheck.values()
    ] == [tokString for tokString in gold_suggestion.values()]
    assert doc._.outcome_spellCheck == gold_outcome
    # splitting components to make use of approx function
    assert [tok.i for tok in doc._.score_spellCheck.keys()] == [
        tok.i for tok in gold_score.keys()
    ]
    assert [tok.text_with_ws for tok in doc._.score_spellCheck.keys()] == [
        tok.text_with_ws for tok in gold_score.keys()
    ]

    assert [
        word_score[0]
        for value in doc._.score_spellCheck.values()
        for word_score in value
    ] == [
        word_score[0] for value in gold_score.values() for word_score in value
    ]
    assert [
        word_score[1]
        for value in doc._.score_spellCheck.values()
        for word_score in value
    ] == approx(
        [
            word_score[1]
            for value in gold_score.values()
            for word_score in value
        ],
        rel=1e-4,
        abs=1e-4,
    )
    nlp.remove_pipe("contextual spellchecker")


def test_span_extensions():
    try:
        nlp.add_pipe(checker)
    except:
        print("contextual SpellCheck already in pipeline")
    doc = nlp(
        "Income was $9.4 milion compared to the prior year of $2.7 milion."
    )

    gold_score = {
        doc[2]: [],
        doc[3]: [],
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
        doc[5]: [],
    }

    assert doc[2:6]._.get_has_spellCheck == True
    # splitting components to make use of approx function
    print(doc[2:6]._.score_spellCheck)
    print(gold_score)
    assert doc[2:6]._.score_spellCheck.keys() == gold_score.keys()
    assert [
        word_score[0]
        for value in doc[2:6]._.score_spellCheck.values()
        for word_score in value
    ] == [
        word_score[0] for value in gold_score.values() for word_score in value
    ]
    assert [
        word_score[1]
        for value in doc[2:6]._.score_spellCheck.values()
        for word_score in value
    ] == approx(
        [
            word_score[1]
            for value in gold_score.values()
            for word_score in value
        ],
        rel=1e-4,
        abs=1e-4,
    )

    # assert doc[2:6]._.score_spellCheck == approx(gold_score,rel=1e-4, abs=1e-4)
    nlp.remove_pipe("contextual spellchecker")


def test_token_extension():
    if "contextual spellchecker" not in nlp.pipe_names:
        nlp.add_pipe(checker)
    doc = nlp(
        "Income was $9.4 milion compared to the prior year of $2.7 milion."
    )

    gold_suggestions = "million"
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
    # Match words and score separately to incorporate approx fn in pytest
    assert [word_score[0] for word_score in doc[4]._.score_spellCheck] == [
        word_score[0] for word_score in gold_score
    ]
    assert [
        word_score[1] for word_score in doc[4]._.score_spellCheck
    ] == approx(
        [word_score[1] for word_score in gold_score], rel=1e-4, abs=1e-4
    )
    nlp.remove_pipe("contextual spellchecker")


def test_warning():
    if "contextual spellchecker" not in nlp.pipe_names:
        nlp.add_pipe(checker)
    merge_ents = nlp.create_pipe("merge_entities")
    nlp.add_pipe(merge_ents)
    doc = nlp(
        "Income was $9.4 milion compared to the prior year of $2.7 milion."
    )

    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        # warnings.simplefilter("always")
        # Trigger a warning.

        assert doc[4]._.get_require_spellCheck == False
        assert doc[4]._.get_suggestion_spellCheck == ""
        assert doc[4]._.score_spellCheck == []
        # Verify Warning
        assert issubclass(w[-1].category, UserWarning)
        assert (
            "Position of tokens modified by downstream element in pipeline eg. merge_entities"
            in str(w[-1].message)
        )

        nlp.remove_pipe("contextual spellchecker")
        print(nlp.pipe_names)

        nlp.remove_pipe("merge_entities")
        print(nlp.pipe_names)
        # warnings.simplefilter("default")

        with pytest.raises(TypeError) as e:
            ContextualSpellCheck(vocab_path=True)
            assert (
                e
                == "Please check datatype provided. vocab_path should be str, debug and performance should be bool"
            )
        max_edit_distance = "non_int_or_float"
        with pytest.raises(ValueError) as e:
            ContextualSpellCheck(max_edit_dist=max_edit_distance)
            assert (
                e
                == f"cannot convert {max_edit_distance} to int. Please provide a valid integer"
            )

    try:
        ContextualSpellCheck(max_edit_dist="3.1")
    except Exception as uncatched_error:
        pytest.fail(str(uncatched_error))


def test_vocab_file():
    with warnings.catch_warnings(record=True) as w:
        ContextualSpellCheck(vocab_path="testing.txt")
        assert any([issubclass(i.category, UserWarning) for i in w])
        assert any(["Using default vocab" in str(i.message) for i in w])
    currentPath = os.path.dirname(__file__)
    debugPathFile = os.path.join(currentPath, "debugFile.txt")
    orgDebugFilePath = os.path.join(currentPath, "originaldebugFile.txt")
    testVocab = os.path.join(currentPath, "testVocab.txt")
    print(testVocab, currentPath, debugPathFile)
    ContextualSpellCheck(vocab_path=testVocab, debug=True)
    with open(orgDebugFilePath) as f1:
        with open(debugPathFile) as f2:
            assert f1.read() == f2.read()


def test_bert_model_name():
    model_name = "a_random_model"
    error_message = (
        f"Can't load config for '{model_name}'. Make sure that:\n\n"
        f"- '{model_name}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
        f"- or '{model_name}' is the correct path to a directory containing a config.json  file\n\n"
    )

    with pytest.raises(OSError) as e:
        ContextualSpellCheck(model_name=model_name)
        assert e == error_message


def test_correct_model_name():
    model_name = "TurkuNLP/bert-base-finnish-cased-v1"
    try:
        ContextualSpellCheck(model_name=model_name)
    except OSError:
        pytest.fail("Specificed model is not present in transformers")
    except Exception as uncatched_error:
        pytest.fail(str(uncatched_error))


@pytest.mark.parametrize(
    "max_edit_distance,expected_spell_check_flag",
    [(0, False), (1, False), (2, True), (3, True)],
)
def test_max_edit_dist(max_edit_distance, expected_spell_check_flag):
    if "contextual spellchecker" in nlp.pipe_names:
        nlp.remove_pipe("contextual spellchecker")
    checker_edit_dist = ContextualSpellCheck(max_edit_dist=max_edit_distance)
    nlp.add_pipe(checker_edit_dist)
    doc = nlp(
        "Income was $9.4 milion compared to the prior year of $2.7 milion."
    )

    # To check the status of `performed_spell_check` flag
    assert doc[4]._.get_require_spellCheck == expected_spell_check_flag
    assert doc[3:5]._.get_has_spellCheck == expected_spell_check_flag
    assert doc._.performed_spellCheck == expected_spell_check_flag

    # To check the response of "suggestions_spellCheck"
    gold_outcome = (
        "Income was $9.4 million compared to the prior year of $2.7 million."
    )
    gold_token = "million"
    gold_outcome = gold_outcome if expected_spell_check_flag else ""
    gold_token = gold_token if expected_spell_check_flag else ""
    print("gold_outcome:", gold_outcome, "gold_token:", gold_token)
    assert doc[4]._.get_suggestion_spellCheck == gold_token
    assert doc._.outcome_spellCheck == gold_outcome

    nlp.remove_pipe("contextual spellchecker")
