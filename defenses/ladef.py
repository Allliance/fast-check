from .base import BaseDefense

KEYWORDS = ["sorry", "I can't", "I can not", "I cannot", "I'm sorry", "I am sorry", "I can't do that", "I can not do that", "I cannot do that"]

class LADefense(BaseDefense):
    def __init__(self,
                 model,
                 max_new_tokens=32,
                 keywords=KEYWORDS,
                 ):
        super().__init__("LADefense", model)
        assert model.use_lade, "Model must be loaded in lookahead mode to use this defense"
        self.keywords = keywords
        self.max_new_tokens = max_new_tokens

    def _is_jailbreak(self, prompt):
        ngrams = self.model(prompt,
                            max_new_tokens=self.max_new_tokens,
                            return_whole_dict=True).ngrams
        
        cnt = 0
        for ngram in ngrams:
            if self.model.tokenizer.encode(self.keywords)[0] in ngram:
                cnt += 1
        
        return cnt / len(ngrams)