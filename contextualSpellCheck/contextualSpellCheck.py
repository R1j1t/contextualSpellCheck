import spacy
import torch
import editdistance
from datetime import datetime
import os
import copy
import warnings, logging

from spacy.tokens import Doc, Token, Span
from spacy.vocab import Vocab

from transformers import AutoModelForMaskedLM, AutoTokenizer


class ContextualSpellCheck(object):
    """
    Class object for Out Of Vocabulary(OOV) corrections
    """

    name = "contextual spellchecker"

    def __init__(
        self,
        vocab_path="",
        model_name="bert-base-cased",
        max_edit_dist=10,
        debug=False,
        performance=False,
    ):
        """To create an object for this class. It does not require any special

        Args:
            vocab_path (str, optional): Vocabulary file path to be used by the
                                         model . Defaults to "".
            model_name (str, optional): Pretrained BERT model name. Defaults to
                                        "bert-base-cased".
            max_edit_dist (int, optional): Maximum edit distance between two
                                           words. Defaults to 10.
            debug (bool, optional): This help prints logs as the data flows
                                     through the class. Defaults to False.
            performance (bool, optional): This is used to print the time taken
                                          by individual steps in spell check.
                                          Defaults to False.
        """
        if (
            (type(vocab_path) != type(""))
            or (type(debug) != type(True))
            or (type(performance) != type(True))
        ):
            raise TypeError(
                "Please check datatype provided. vocab_path should be str,"
                " debug and performance should be bool"
            )
        try:
            int(float(max_edit_dist))
        except ValueError as identifier:
            raise ValueError(
                f"cannot convert {max_edit_dist} to int. Please provide a valid integer"
            )

        if vocab_path != "":
            try:
                # First open() for user specified word addition to vocab
                with open(vocab_path, encoding="utf8") as f:
                    # if want to remove '[unusedXX]' from vocab
                    # words = [
                    #     line.rstrip()
                    #     for line in f
                    #     if not line.startswith("[unused")
                    # ]
                    words = [line.strip() for line in f]

                # The below code adds the necessary words like numbers
                # /punctuations/tokenizer specific words like [PAD]/[
                # unused0]/##M
                current_path = os.path.dirname(__file__)
                vocab_path = os.path.join(current_path, "data", "vocab.txt")
                extra_token = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
                words.extend(extra_token)

                with open(vocab_path, encoding="utf8") as f:
                    # if want to remove '[unusedXX]' from vocab
                    # words = [
                    #     line.rstrip()
                    #     for line in f
                    #     if not line.startswith("[unused")
                    # ]
                    for line in f:
                        extra_token = line.strip()
                        if extra_token.startswith("[unused"):
                            words.append(extra_token)
                        elif extra_token.startswith("##"):
                            words.append(extra_token)
                        elif len(extra_token) == 1:
                            words.append(extra_token)
                if debug:
                    debug_file_path = os.path.join(
                        current_path, "tests", "debugFile.txt"
                    )
                    with open(debug_file_path, "w+") as new_file:
                        new_file.write("\n".join(words))
                    print("Final vocab at " + debug_file_path)

            except Exception as e:
                print(e)
                warnings.warn("Using default vocab")
                vocab_path = ""
                words = []

        self.max_edit_dist = int(float(max_edit_dist))
        self.model_name = model_name
        self.BertTokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if vocab_path == "":
            words = list(self.BertTokenizer.get_vocab().keys())
        self.vocab = Vocab(strings=words)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        self.BertModel = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.mask = self.BertTokenizer.mask_token
        self.debug = debug
        self.performance = performance
        if not Doc.has_extension("contextual_spellCheck"):
            Doc.set_extension("contextual_spellCheck", default=True)
            Doc.set_extension("performed_spellCheck", default=False)

            Doc.set_extension("suggestions_spellCheck", default={})
            Doc.set_extension("outcome_spellCheck", default="")
            Doc.set_extension("score_spellCheck", default=None)

            Span.set_extension(
                "get_has_spellCheck", getter=self.span_require_spell_check
            )
            Span.set_extension(
                "score_spellCheck", getter=self.span_score_spell_check
            )

            Token.set_extension(
                "get_require_spellCheck", getter=self.token_require_spell_check
            )
            Token.set_extension(
                "get_suggestion_spellCheck",
                getter=self.token_suggestion_spell_check,
            )
            Token.set_extension(
                "score_spellCheck", getter=self.token_score_spell_check
            )

    def __call__(self, doc):
        """
        call function for the class. Used in spacy pipeline

        Args:
            doc {`Spacy.Doc`}: Spacy Doc Object

        Returns:
            `Spacy.Doc`: Updated doc object with custom extensions values
        """
        if self.performance:
            model_loaded = datetime.now()
        misspell_tokens, doc = self.misspell_identify(doc)
        if self.performance:
            self.time_log("Misspell identification: ", model_loaded)
        if len(misspell_tokens) > 0:
            doc, candidate = self.candidate_generator(doc, misspell_tokens)
            if self.performance:
                self.time_log("candidate Generator: ", model_loaded)
            self.candidate_ranking(doc, candidate)
            if self.performance:
                self.time_log("candidate ranking: ", model_loaded)
        return doc

    def check(self, query="", spacy_model="en_core_web_sm"):
        """
        Complete pipeline for **testing purpose only**

        Keyword Args:
            query (str): query for which spell check model to run
                            (default: {""})
            spacy_model (str): Name of spacy model

        Returns:
            (str, `Doc`): returns updated query (if no oov words then "")
                          and updated Doc Object
        """
        if type(query) != str and len(query) == 0:
            return "Invalid query, expected non empty `str` but passed", query

        nlp = spacy.load(spacy_model, disable=["tagger", "parser"])
        doc = nlp(query)
        model_loaded = datetime.now()
        misspell_tokens, doc = self.misspell_identify(doc)
        self.time_log("Misspell identification: ", model_loaded)
        update_query = ""
        if len(misspell_tokens) > 0:
            candidate = self.candidate_generator(doc, misspell_tokens)
            answer = self.candidate_ranking(candidate)
            for i in doc:
                if i in misspell_tokens:
                    update_query += answer[i] + i.whitespace_
                else:
                    update_query += i.text_with_ws

            print("Did you mean: ", update_query)
            doc._.set("outcome_spellCheck", update_query)
            # problem with below as it modifies the original object
        #             with doc.retokenize() as retokenizer:
        #                 print("Original text:",retokenizer.merge(doc[:]))
        return update_query, doc

    def misspell_identify(self, doc, query=""):
        """To identify misspelled words from the query

        At present, All the following criteria should be met for word to be
        misspelled
        1. Should not be in our vocab
        2. should not be a Person
        3. Should not be a number
        4. Should not be a url
        5. Should not be a space
        6. Should not be punctuation
        7. Should not be a Geopolitical Entity
        8. Should not be a Organisation

        Args:
            doc {`Spacy.Doc`}: Spacy doc object as input

        Keyword Args:
            query {str}: not used now (default: {""})

        Returns:
            `tuple`: returns `List[`Spacy.Token`]` and `Spacy.Doc`
        """

        # deep copy is required to preserve individual token info
        # from objects in pipeline which can modify token info
        # like merge_entities
        docCopy = copy.deepcopy(doc)

        misspell = []
        for token in docCopy:
            if (
                (token.text.lower() not in self.vocab)
                and (token.ent_type_ != "PERSON")
                and (not token.like_num)
                and (not token.like_email)
                and (not token.like_url)
                # added after 0.0.4
                and (not token.is_space)
                and (not token.is_punct)
                and (token.ent_type_ != "GPE")
                and (token.ent_type_ != "ORG")
            ):
                misspell.append(token)

        if self.debug:
            print("misspell identified: ", misspell)
        return misspell, doc

    def candidate_generator(self, doc, misspellings, top_n=10):
        """Returns Candidates for misspell words

        This function is responsible for generating candidate list for misspell
        using BERT. The misspell is masked with a token (eg [MASK]) and the
        model tries to predict `n` candidates for that mask. The `doc` is used
         to provide sentence (context) for the mask


        Args:
            doc {`Spacy.Doc`}: Spacy Doc object, used to provide context to
                               the model misspellings
           {List(`Spacy.Token`)}: Contains List of `Token` object types from
                                  spacy to preserve meta information of the
                                  token

        Keyword Args:
            top_n {int}:  # suggestions to be considered (default: {10})

        Returns:
            Dict{`Token`:List[{str}]}: Eg of return type {misspell-1:
                                      ['candidate-1','candidate-2', ...],
                                      misspell-2:['candidate-1','candidate-2'
                                      . ...]}
        """
        response = {}
        score = {}

        for token in misspellings:
            update_query = ""
            # Instead of using complete doc, we use sentence to provide context
            # and improve performance
            if self.debug:
                print(token.text, token.sent)
            for i in token.sent:
                if i.i == token.i:
                    update_query += self.mask + i.whitespace_
                else:
                    update_query += i.text_with_ws
            if self.debug:
                print(
                    "\nFor",
                    "`" + token.text + "`",
                    "updated query is:\n",
                    update_query,
                )

            model_input = self.BertTokenizer.encode(
                update_query, return_tensors="pt"
            )
            mask_token_index = torch.where(
                model_input == self.BertTokenizer.mask_token_id
            )[1]
            token_logits = self.BertModel(model_input)[0]
            mask_token_logits = token_logits[0, mask_token_index, :]
            token_probability = torch.nn.functional.softmax(
                mask_token_logits, dim=1
            )
            top_n_score, top_n_tokens = torch.topk(
                token_probability, top_n, dim=1
            )
            top_n_tokens = top_n_tokens[0].tolist()
            top_n_score = top_n_score[0].tolist()
            if self.debug:
                # print("top_n_tokens:", top_n_tokens)
                print("token_score: ", top_n_score)

            if token not in response:
                response[token] = [
                    self.BertTokenizer.decode([candidateWord])
                    for candidateWord in top_n_tokens
                ]
                score[token] = [
                    (
                        self.BertTokenizer.decode([top_n_tokens[i]]),
                        round(top_n_score[i], 5),
                    )
                    for i in range(top_n)
                ]

            if self.debug:
                print(
                    "response[" + "`" + str(token) + "`" + "]: ",
                    response[token],
                    "score[" + "`" + str(token) + "`" + "]: ",
                    score[token],
                )

        if len(misspellings) != 0:
            doc._.set("performed_spellCheck", True)
            doc._.set("score_spellCheck", score)

        return doc, response

    def candidate_ranking(self, doc, misspellings_dict):
        """Ranking the candidates based on edit Distance

        At present using a library to calculate edit distance
        between actual word and candidate words. Candidate word
        for which edit distance is lowest is selected. If least
        edit distance is same then word with higher probability
        is selected by default

        Args:
            misspellingsDict {Dict{`Token`:List[{str}]}}:
            Original token is the key and candidate words are the values

        Returns:
            Dict{`Token`:{str}}: Eg of return type {misspell-1:'BEST-CANDIDATE'}
        """

        response = {}
        #         doc = self.nlp(query)
        for misspell in misspellings_dict:
            # Init least_edit distance
            least_edit_dist = self.max_edit_dist

            if self.debug:
                print(
                    "misspellings_dict[misspell]", misspellings_dict[misspell]
                )
            for candidate in misspellings_dict[misspell]:
                edit_dist = editdistance.eval(misspell.text, candidate)
                if edit_dist < least_edit_dist:
                    least_edit_dist = edit_dist
                    response[misspell] = candidate

            if self.debug:
                if len(response) != 0:
                    print(
                        "response[" + "`" + str(misspell) + "`" + "]",
                        response[misspell],
                    )
                else:
                    print(
                        "No candidate selected for max_edit_dist="
                        + str(self.max_edit_dist)
                    )

        if len(response) > 0:
            doc._.set("suggestions_spellCheck", response)
            update_query = ""
            for i in doc:
                update_token = i.text_with_ws
                for misspell in response.keys():
                    if i.i == misspell.i:
                        update_token = response[misspell] + misspell.whitespace_
                        break
                update_query += update_token
            doc._.set("outcome_spellCheck", update_query)
        else:
            doc._.set("performed_spellCheck", False)

        if self.debug:
            print("Final suggestions", doc._.suggestions_spellCheck)

        return response

    @staticmethod
    def time_log(fn_name, relative_time):
        """For time log

        Args:
            fnName {str}: function name to print
            relativeTime {datetime}: previous date time for subtraction

        Returns:
            datetime: datetime of current logging
        """

        time_now = datetime.now()
        print(fn_name, "took: ", time_now - relative_time)
        return datetime.now()

    @staticmethod
    def token_require_spell_check(token):
        """Getter for Token attributes.
        Args:
            token {`Spacy.Token`}: Token object for the value should be returned
        Returns:
            List: If no suggestions: False else: True
        """
        return any(
            [
                token.i == suggestion.i and token.text == suggestion.text
                for suggestion in token.doc._.suggestions_spellCheck.keys()
            ]
        )

    @staticmethod
    def token_suggestion_spell_check(token):
        """
        Getter for Token attributes.

        Args:
            token {`Spacy.Token`}: Token object for the value should be returned

        Returns:
            List: If no suggestions: [] else: List['suggestion-1','
                  suggestion-1',...]
        """
        for suggestion in token.doc._.suggestions_spellCheck.keys():
            if token.i == suggestion.i:
                if token.text_with_ws == suggestion.text_with_ws:
                    return token.doc._.suggestions_spellCheck[suggestion]
                else:
                    warnings.warn(
                        "Position of tokens modified by downstream element "
                        "in pipeline eg. merge_entities"
                    )
        return ""

    @staticmethod
    def token_score_spell_check(token):
        """
        Getter for Token attributes.

        Args:
            token {`Spacy.Token`} :Token object for the value should be returned

        Returns:
            List :If no suggestions: [] else: List[('suggestion-1',score-1),
                 ('suggestion-1',score-2), ...]
        """
        if token.doc._.score_spellCheck is None:
            return []
        for suggestion in token.doc._.score_spellCheck.keys():
            if token.i == suggestion.i:
                if token.text == suggestion.text:
                    return token.doc._.score_spellCheck[suggestion]
                else:
                    warnings.warn(
                        "Position of tokens modified by downstream element"
                        " in pipeline eg. merge_entities"
                    )
        return []

    def span_score_spell_check(self, span):
        """
        Getter for Span Object

        Args:
            span {`Spacy.Span`} :Span object for which value should be returned

        Returns:
            Dict(`Token`:List(str,int)) :for every token it will return
                                        (suggestion,score) eg: {token-1: [],
                                        token-2: [], token-3: [('suggestion-1',
                                        score-1), ...], ...}
        """
        return {token: self.token_score_spell_check(token) for token in span}

    def span_require_spell_check(self, span):
        """
        Getter for Span Object

        Args:
            span {`Spacy.Span`} :Span object for which value should be returned

        Returns:
            Boolean :True if the span requires spellCheck
        """
        return any([self.token_require_spell_check(token) for token in span])

    @staticmethod
    def doc_suggestions_spell_check(doc):
        """
        Getter for Doc attribute

        Args:
            doc {`Spacy.Doc`} :Doc object for which value should be returned

        Returns:
            Dict(`Spacy.Token`:List(str)) :{misspell-1: ['suggestion-1',
                                            'suggestion-2'...]}
        """
        response = {}
        if doc._.score_spellCheck is None:
            return response
        for token in doc._.score_spellCheck:
            if token not in response:
                response[token] = []
            for suggestion_score in doc._.score_spellCheck[token]:
                response[token].append(suggestion_score[0])
        return response

    def doc_outcome_spell_check(self, doc):
        """
        Getter for Doc attribute

        Args:
            doc {`Spacy.Doc`} :Doc object for which value should be returned

        Returns:
            str :updated sentence
        """
        if not doc._.performed_spellCheck:
            return ""

        update_query = ""
        suggestions = doc._.suggestions_spellCheck

        for i in doc:
            update_token = i.text_with_ws
            for misspell in suggestions.keys():
                if misspell.text_with_ws in i.text_with_ws:
                    update_token = suggestions[misspell] + misspell.whitespace_
                    suggestions.remove(misspell)
                    break

            update_query += update_token

        if self.debug:
            print("Did you mean: ", update_query)

        return update_query


if __name__ == "__main__":
    print("Code running...")
    nlp = spacy.load("en_core_web_sm")
    # for issue #1
    # merge_ents = nlp.create_pipe("merge_entities")
    if "parser" not in nlp.pipe_names:
        raise AttributeError(
            "parser is required please enable it in nlp pipeline"
        )
    checker = ContextualSpellCheck(debug=True, max_edit_dist=3)
    nlp.add_pipe(checker)
    # nlp.add_pipe(merge_ents)

    doc = nlp(
        "Income was $9.4 milion compared to the prior year of $2.7 milion."
    )

    print("=" * 20, "Doc Extension Test", "=" * 20)
    print(doc._.outcome_spellCheck)

    print(doc._.contextual_spellCheck)
    print(doc._.performed_spellCheck)
    print(doc._.suggestions_spellCheck)
    print(doc._.score_spellCheck)

    token_pos = 4
    print("=" * 20, "Token Extension Test", "=" * 20)
    print(doc[token_pos].text, doc[token_pos].i)
    print(doc[token_pos]._.get_require_spellCheck)
    print(doc[token_pos]._.get_suggestion_spellCheck)
    print(doc[token_pos]._.score_spellCheck)

    span_start = token_pos - 2
    span_end = token_pos + 2
    print("=" * 20, "Span Extension Test", "=" * 20)
    print(doc[span_start:span_end].text)
    print(doc[span_start:span_end]._.get_has_spellCheck)
    print(doc[span_start:span_end]._.score_spellCheck)
