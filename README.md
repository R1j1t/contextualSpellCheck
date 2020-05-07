# spellCheck

Contextual word checker for better suggestions

## Types of spelling mistakes:

> Spelling errors are broadly classified as non- word errors (NWE) and real word errors (RWE). If the misspelt string is a valid word in the language, then it is called an RWE, else it is an NWE.
>
> -- [Monojit Choudhury et. al. (2007)][1]

## API

At present, there is a get API in a flask app. You can run the app and expect the following output from the API.

```json
{
    "success": true,
    "input": "Income was $9.4 milion compared to the prior year of $2.7 milion.",
    "corrected": "Income was $9.4 million compared to the prior year of $2.7 million.",
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

## Reference

[1]: <http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=52A3B869596656C9DA285DCE83A0339F?doi=10.1.1.146.4390&rep=rep1&type=pdf>
