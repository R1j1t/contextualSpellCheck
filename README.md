# spellCheck
<a href="https://github.com/R1j1t/contextualSpellCheck"><img src="https://user-images.githubusercontent.com/22280243/82138959-2852cd00-9842-11ea-918a-49b2a7873ef6.png" width="276" height="120" align="right" /></a>

Contextual word checker for better suggestions

[![license](https://img.shields.io/github/license/r1j1t/contextualSpellCheck)](https://github.com/R1j1t/contextualSpellCheck/blob/master/LICENSE) 
[![PyPI](https://img.shields.io/pypi/v/contextualSpellCheck?color=green)](https://pypi.org/project/contextualSpellCheck/) 
[![Python-Version](https://img.shields.io/badge/Python-3.6+-green)](https://github.com/R1j1t/contextualSpellCheck#install)
[![Downloads](https://pepy.tech/badge/contextualspellcheck/week)](https://pepy.tech/project/contextualspellcheck)
[![GitHub contributors](https://img.shields.io/github/contributors/r1j1t/contextualSpellCheck)](https://github.com/R1j1t/contextualSpellCheck/graphs/contributors)
[![Help Wanted](https://img.shields.io/badge/Help%20Wanted-Task%20List-violet)](https://github.com/R1j1t/contextualSpellCheck#task-list)
[![DOI](https://zenodo.org/badge/254703118.svg)](https://zenodo.org/badge/latestdoi/254703118)

## Types of spelling mistakes

It is essential to understand that identifying whether a candidate is a spelling error is a big task.

> Spelling errors are broadly classified as non- word errors (NWE) and real word errors (RWE). If the misspelt string is a valid word in the language, then it is called an RWE, else it is an NWE.
>
> -- [Monojit Choudhury et. al. (2007)][1]

This package currently focuses on Out of Vocabulary (OOV) word or non-word error (NWE) correction using BERT model. The idea of using BERT was to use the context when correcting OOV. To improve this package, I would like to extend the functionality to identify RWE, optimising the package, and improving the documentation.

## Install

The package can be installed using [pip](https://pypi.org/project/contextualSpellCheck/). You would require python 3.6+

```bash
pip install contextualSpellCheck
```

## Usage

**Note:** For use in other languages check [`examples`](https://github.com/R1j1t/contextualSpellCheck/tree/master/examples) folder.

### How to load the package in spacy pipeline

```python
>>> import contextualSpellCheck
>>> import spacy
>>> nlp = spacy.load("en_core_web_sm") 
>>> 
>>> ## We require NER to identify if a token is a PERSON
>>> ## also require parser because we use `Token.sent` for context
>>> nlp.pipe_names
['tok2vec', 'tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer']
>>> contextualSpellCheck.add_to_pipe(nlp)
>>> nlp.pipe_names
['tok2vec', 'tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer', 'contextual spellchecker']
>>> 
>>> doc = nlp('Income was $9.4 milion compared to the prior year of $2.7 milion.')
>>> doc._.outcome_spellCheck
'Income was $9.4 million compared to the prior year of $2.7 million.'
```

Or you can add to spaCy pipeline manually!

```python
>>> import spacy
>>> import contextualSpellCheck
>>> 
>>> nlp = spacy.load("en_core_web_sm")
>>> nlp.pipe_names
['tok2vec', 'tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer']
>>> # You can pass the optional parameters to the contextualSpellCheck
>>> # eg. pass max edit distance use config={"max_edit_dist": 3}
>>> nlp.add_pipe("contextual spellchecker")
<contextualSpellCheck.contextualSpellCheck.ContextualSpellCheck object at 0x1049f82b0>
>>> nlp.pipe_names
['tok2vec', 'tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer', 'contextual spellchecker']
>>> 
>>> doc = nlp("Income was $9.4 milion compared to the prior year of $2.7 milion.")
>>> print(doc._.performed_spellCheck)
True
>>> print(doc._.outcome_spellCheck)
Income was $9.4 million compared to the prior year of $2.7 million.
```

After adding `contextual spellchecker` in the pipeline, you use the pipeline normally. The spell check suggestions and other data can be accessed using [extensions](#Extensions).

### Using the pipeline

```python
>>> doc = nlp(u'Income was $9.4 milion compared to the prior year of $2.7 milion.')
>>> 
>>> # Doc Extention
>>> print(doc._.contextual_spellCheck)
True
>>> print(doc._.performed_spellCheck)
True
>>> print(doc._.suggestions_spellCheck)
{milion: 'million', milion: 'million'}
>>> print(doc._.outcome_spellCheck)
Income was $9.4 million compared to the prior year of $2.7 million.
>>> print(doc._.score_spellCheck)
{milion: [('million', 0.59422), ('billion', 0.24349), (',', 0.08809), ('trillion', 0.01835), ('Million', 0.00826), ('%', 0.00672), ('##M', 0.00591), ('annually', 0.0038), ('##B', 0.00205), ('USD', 0.00113)], milion: [('billion', 0.65934), ('million', 0.26185), ('trillion', 0.05391), ('##M', 0.0051), ('Million', 0.00425), ('##B', 0.00268), ('USD', 0.00153), ('##b', 0.00077), ('millions', 0.00059), ('%', 0.00041)]}
>>> 
>>> # Token Extention
>>> print(doc[4]._.get_require_spellCheck)
True
>>> print(doc[4]._.get_suggestion_spellCheck)
'million'
>>> print(doc[4]._.score_spellCheck)
[('million', 0.59422), ('billion', 0.24349), (',', 0.08809), ('trillion', 0.01835), ('Million', 0.00826), ('%', 0.00672), ('##M', 0.00591), ('annually', 0.0038), ('##B', 0.00205), ('USD', 0.00113)]
>>> 
>>> # Span Extention
>>> print(doc[2:6]._.get_has_spellCheck)
True
>>> print(doc[2:6]._.score_spellCheck)
{$: [], 9.4: [], milion: [('million', 0.59422), ('billion', 0.24349), (',', 0.08809), ('trillion', 0.01835), ('Million', 0.00826), ('%', 0.00672), ('##M', 0.00591), ('annually', 0.0038), ('##B', 0.00205), ('USD', 0.00113)], compared: []}
```

## Extensions

To make the usage easy, `contextual spellchecker` provides custom spacy extensions which your code can consume. This makes it easier for the user to get the desired data. contextualSpellCheck provides extensions on the `doc`, `span` and `token` level. The below tables summarise the extensions.

### `spaCy.Doc` level extensions

| Extension | Type | Description | Default |
|------------------------------|---------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|---------|
| doc._.contextual_spellCheck | `Boolean` | To check whether contextualSpellCheck is added as extension | `True` |
| doc._.performed_spellCheck | `Boolean` | To check whether contextualSpellCheck identified any misspells and performed correction | `False` |
| doc._.suggestions_spellCheck | `{Spacy.Token:str}` | if corrections are performed, it returns the mapping of misspell token (`spaCy.Token`) with suggested word(`str`) | `{}` |
| doc._.outcome_spellCheck | `str` | corrected sentence(`str`) as output | `""` |
| doc._.score_spellCheck | `{Spacy.Token:List(str,float)}` | if corrections are identified, it returns the mapping of misspell token (`spaCy.Token`) with suggested words(`str`) and probability of that correction | `None` |

### `spaCy.Span` level extensions
| Extension | Type | Description | Default |
|-------------------------------|---------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|
| span._.get_has_spellCheck | `Boolean` | To check whether contextualSpellCheck identified any misspells and performed correction in this span | `False` |
| span._.score_spellCheck | `{Spacy.Token:List(str,float)}` | if corrections are identified, it returns the mapping of misspell token (`spaCy.Token`) with suggested words(`str`) and probability of that correction for tokens in this `span` | `{spaCy.Token: []}` |

### `spaCy.Token` level extensions

| Extension | Type | Description | Default |
|-----------------------------------|-----------------|-------------------------------------------------------------------------------------------------------------|---------|
| token._.get_require_spellCheck | `Boolean` | To check whether contextualSpellCheck identified any misspells and performed correction on this `token` | `False` |
| token._.get_suggestion_spellCheck | `str` | if corrections are performed, it returns the suggested word(`str`) | `""` |
| token._.score_spellCheck | `[(str,float)]` | if corrections are identified, it returns suggested words(`str`) and probability(`float`) of that correction | `[]` |

## API

At present, there is a simple GET API to get you started. You can run the app in your local and play with it.

Query: You can use the endpoint http://127.0.0.1:5000/?query=YOUR-QUERY
Note: Your browser can handle the text encoding

```
GET Request: http://localhost:5000/?query=Income%20was%20$9.4%20milion%20compared%20to%20the%20prior%20year%20of%20$2.7%20milion.
```

Response:

```json
{
    "success": true,
    "input": "Income was $9.4 milion compared to the prior year of $2.7 milion.",
    "corrected": "Income was $9.4 milion compared to the prior year of $2.7 milion.",
    "suggestion_score": {
        "milion": [
            [
                "million",
                0.59422
            ],
            [
                "billion",
                0.24349
            ],
            ...
        ],
        "milion:1": [
            [
                "billion",
                0.65934
            ],
            [
                "million",
                0.26185
            ],
            ...
        ]
    }
}
```

## Task List

- [ ] use cython for part of the code to improve performance ([#39](https://github.com/R1j1t/contextualSpellCheck/issues/39))
- [ ] Improve metric for candidate selection ([#40](https://github.com/R1j1t/contextualSpellCheck/issues/40))
- [ ] Add examples for other langauges ([#41](https://github.com/R1j1t/contextualSpellCheck/issues/41))
- [ ] Update the logic of misspell identification (OOV) ([#44](https://github.com/R1j1t/contextualSpellCheck/issues/44))
- [ ] better candidate generation (solved by [#44](https://github.com/R1j1t/contextualSpellCheck/issues/44)?)
- [ ] add metric by testing on datasets
- [ ] Improve documentation
- [ ] Improve logging in code
- [ ] Add support for Real Word Error (RWE) (Big Task)
- [ ] add multi mask out capability

<details><summary>Completed Task</summary>
<p>

- [x] specify maximum edit distance for `candidateRanking`
- [x] allow user to specify bert model
- [x] Include transformers deTokenizer to get better suggestions
- [x] dependency version in setup.py ([#38](https://github.com/R1j1t/contextualSpellCheck/issues/38))

</p>
</details>

## Support and contribution

If you like the project, please ⭑ the project and show your support! Also, if you feel, the current behaviour is not as expected, please feel free to raise an [issue](https://github.com/R1j1t/contextualSpellCheck/issues). If you can help with any of the above tasks, please open a [PR](https://github.com/R1j1t/contextualSpellCheck/pulls) with necessary changes to documentation and tests.

## Cite

If you are using contextualSpellCheck in your academic work, please consider citing the library using the below BibTex entry:

```bibtex
@misc{Goel_Contextual_Spell_Check_2021,
author = {Goel, Rajat},
doi = {10.5281/zenodo.4642379},
month = {3},
title = {{Contextual Spell Check}},
url = {https://github.com/R1j1t/contextualSpellCheck},
year = {2021}
}
```



## Reference

Below are some of the projects/work I referred to while developing this package

1. Explosion AI.Architecture. May 2020. url:https://spacy.io/api.
2. Monojit Choudhury et al. “How difficult is it to develop a perfect spell-checker? A cross-linguistic analysis through complex network approach”. In:arXiv preprint physics/0703198(2007).
3. Jacob Devlin et al. BERT: Pre-training of Deep Bidirectional Transform-ers for Language Understanding. 2019. arXiv:1810.04805 [cs.CL].
4. Hugging  Face.Fast Coreference Resolution in spaCy with Neural Net-works. May 2020. url:https://github.com/huggingface/neuralcoref.
5. Ines.Chapter 3: Processing Pipelines. May 20202. url:https://course.spacy.io/en/chapter3.
6. Eric Mays, Fred J Damerau, and Robert L Mercer. “Context based spellingcorrection”. In:Information Processing & Management27.5 (1991), pp. 517–522.
7. Peter Norvig. How to Write a Spelling Corrector. May 2020. url:http://norvig.com/spell-correct.html.
8. Yifu  Sun  and  Haoming  Jiang.Contextual Text Denoising with MaskedLanguage Models. 2019. arXiv:1910.14080 [cs.CL].
9. Thomas  Wolf  et  al.  “Transformers:  State-of-the-Art  Natural  LanguageProcessing”. In:Proceedings of the 2020 Conference on Empirical Methodsin Natural Language Processing: System Demonstrations. Online: Associ-ation for Computational Linguistics, Oct. 2020, pp. 38–45. url:https://www.aclweb.org/anthology/2020.emnlp-demos.6.

[1]: <http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=52A3B869596656C9DA285DCE83A0339F?doi=10.1.1.146.4390&rep=rep1&type=pdf>
