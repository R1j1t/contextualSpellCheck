import spacy
import torch
import editdistance
import datetime

from spacy.tokens import Token
from spacy.vocab import Vocab

from transformers import AutoModelWithLMHead, AutoTokenizer


class oovChecker:
    """Class object for Out Of Vocabulary(OOV) corrections 
    """

    def __init__(self, debug=False):
        self.nlp = spacy.load(
            "en_core_web_sm", disable=["tagger", "parser"]
        )  # using default tokeniser with NER
        with open("./uncased_L-4_H-512_A-8/vocab.txt") as f:
            # if want to remove '[unusedXX]' from vocab
            # words = [line.rstrip() for line in f if not line.startswith('[unused')]
            words = [line.rstrip() for line in f]
        self.vocab = Vocab(strings=words)
        self.BertTokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.BertModel = AutoModelWithLMHead.from_pretrained("bert-base-cased")
        self.mask = self.BertTokenizer.mask_token
        self.debug = debug

    def check(self, query=""):
        """Complete pipeline which returns update query

        Keyword Arguments:
            query {str} -- User query for which spell checking to be done (default: {''})

        Returns:
            {str} -- returns updated query with spelling corrections (if any)
        """
        if type(query) != str and len(query) == 0:
            print("Invalid query, expected non empty `str` but passed", query)

        misspellTokens, doc = self.misspellIdentify(query)
        if len(misspellTokens) > 0:
            candidate = self.candidateGenerator(misspellTokens, query=query)
            answer = self.candidateRanking(candidate)
            updatedQuery = ""
            for i in doc:
                if i in misspellTokens:
                    updatedQuery += answer[i] + " "
                else:
                    updatedQuery += i.text + " "

            print("Did you mean: ", updatedQuery)
            print("Original text:", query)
        return updatedQuery

    def misspellIdentify(self, query=""):
        """To identify misspelled words from the query

        At present, All the following criteria should be met for word to be misspelled
        1. Should not in our vocab
        2. should not be a Person
        3. Should not be a number


        Keyword Arguments:
            query {str} -- user query eg: "aa bb cc..." (default: {''})

        Returns:
            {tuple} -- returns `List[`Token`]` and `Doc`
        """

        doc = self.nlp(query)
        misspell = []
        for token in doc:
            if (
                (token.text.lower() not in self.vocab)
                and (token.ent_type_ != "PERSON")
                and (not token.like_num)
            ):

                misspell.append(token)

        if self.debug:
            print(misspell)
        return (misspell, doc)

    def candidateGenerator(self, misspellings, top_n=5, query=""):
        """Returns Candidates for misspells

        This function is responsible for generating candidate list for misspell
        using BERT. The misspell is masked with a token and the model tries to 
        predict `n` candidates for the mask.

        Arguments:
            misspellings {List[`Token`]} -- Contains List of `Token` object types 
            from spacy to preserve meta information of the token 

        Keyword Arguments:
            top_n {int} -- Number of candidates to be generated (default: {5})
            query {User query} -- This is used for context pwered candidate generations.  (default: {''})

        Returns:
            Dict{`Token`:List[{str}]} -- Eg of return type {misspell-1:['candidate-1','candidate-2', ...],
                            misspell-2:['candidate-1','candidate-2'. ...]}
        """

        response = {}

        for token in misspellings:
            updatedQuery = query
            updatedQuery = updatedQuery.replace(token.text, self.mask)
            if self.debug:
                print(
                    "For", "`" + token.text + "`", "updated query is:\n", updatedQuery
                )

            model_input = self.BertTokenizer.encode(updatedQuery, return_tensors="pt")
            mask_token_index = torch.where(
                model_input == self.BertTokenizer.mask_token_id
            )[1]
            token_logits = self.BertModel(model_input)[0]
            mask_token_logits = token_logits[0, mask_token_index, :]

            top_n_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
            if self.debug:
                print("top_n_tokens:", top_n_tokens)

            if token not in response:
                response[token] = [
                    self.BertTokenizer.decode([candidateWord])
                    for candidateWord in top_n_tokens
                ]

            # for candidate in top_5_tokens:
            # response[token].append(self.BertTokenizer.decode([candidate]))
            # print(updatedQuery.replace(self.mask, self.BertTokenizer.decode([candidate])))

            if self.debug:
                print(response)

        return response

    def candidateRanking(self, misspellingsDict):
        """Ranking the candidates based on edit Distance

        At present using a library to calculate edit distance 
        between actual word and candidate words. Candidate word 
        for which edit distance is lowest is selected. If least 
        edit distance is same then word with higher probability 
        is selected by default

        Arguments:
            misspellingsDict {Dict{`Token`:List[{str}]}} -- 
            Orginal token is the key and candidate words are the values 

        Returns:
            Dict{`Token`:{str}} -- Eg of return type {misspell-1:'BEST-CANDIDATE'}
        """

        response = {}
        #         doc = self.nlp(query)
        for misspell in misspellingsDict:
            ## Init least_edit distance
            least_edit_dist = 100

            if self.debug:
                print("misspellingsDict[misspell]", misspellingsDict[misspell])
            for candidate in misspellingsDict[misspell]:
                edit_dist = editdistance.eval(misspell.text, candidate)
                if edit_dist < least_edit_dist:
                    least_edit_dist = edit_dist
                    response[misspell] = candidate

            if self.debug:
                print(response)
        return response


def timeLog(fnName, relativeTime):
    """For time log

    Arguments:
        fnName {str} -- function name to print
        relativeTime {datetime} -- previous date time for subtraction

    Returns:
        datetime -- datetime of current logging
    """

    timeNow = datetime.datetime.now()
    print(fnName, "took: ", timeNow - relativeTime)
    return datetime.datetime.now()


if __name__ == "__main__":
    print("Code running...")
    start = datetime.datetime.now()
    checker = oovChecker()
    modelLoadTime = timeLog("Model Loading", start)

    checker.check("Income was $9.4 million compared to the prior year of $2.7 milion.")
    checkerTime = timeLog("Correction (if any)", modelLoadTime)
