from flask import request
from contextualSpellCheck.contextualSpellCheck import contextualSpellCheck
import spacy
import json
from flask import Flask, render_template, request, make_response, jsonify


# url
# http://10.1.1.1:5000/login/alex

app = Flask(__name__)

nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser"])
checker = contextualSpellCheck(debug=False)
nlp.add_pipe(checker)


@app.route("/", methods=["GET"])
def model_query():
    query = request.args.get("query")

    doc = nlp(query)
    correctionScore = {}
    duplicate_counter = 1
    for key, value in doc._.score_spellCheck.items():
        if key.text in correctionScore:
            key = key.text + ":" + str(duplicate_counter)
            duplicate_counter += 1
        else:
            key = key.text
            duplicate_counter = 1
        correctionScore[key] = value

    with doc.retokenize() as retokenizer:
        retokenizer.merge(doc[0:])

    response = make_response(
        json.dumps(
            {
                "success": True,
                "input": doc.text,
                "corrected": doc._.outcome_spellCheck,
                "suggestion_score": correctionScore,
            }
        )
    )
    response.status_code = 200
    response.headers["Content-Type"] = "application/json"
    return response


if __name__ == "__main__":
    app.run()
