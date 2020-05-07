from flask import request
from oovSpellCheck import spellChecker
import spacy
import json
from flask import Flask, render_template, request, make_response, jsonify


#url
# http://10.1.1.1:5000/login/alex

app = Flask(__name__)

nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser"])
checker = spellChecker(debug=False)
nlp.add_pipe(checker)
        
    
@app.route('/', methods=['GET'])
def model_query():
    query = request.args.get('query')

    doc = nlp(query)
    response = make_response(json.dumps({'success': True, 'corrected': doc._.outcome_spellCheck}))
    response.status_code = 200
    response.headers['Content-Type'] = 'application/json'
    return response

if __name__ == "__main__":
	app.run()
