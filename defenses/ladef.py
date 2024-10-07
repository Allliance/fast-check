from .base import BaseDefense
import numpy as np

KEYWORDS = ["sorry", "I can't", "I can not", "I cannot", "I'm sorry", "I am sorry", "I can't do that", "I can not do that", "I cannot do that"]

class LADefense(BaseDefense):
    def __init__(self,
                 model,
                 max_new_tokens=32,
                 keywords=KEYWORDS,
                 **kwargs,
                 ):
        super().__init__("LADefense", model)
        assert model.use_lade, "Model must be loaded in lookahead mode to use this defense"
        self.keywords = keywords
        self.max_new_tokens = max_new_tokens

    def _is_jailbreak(self, prompt):        
        ngrams = self.llm(prompt,
                            max_new_tokens=80,
                            return_ngrams=True)[1]
        
        cnt = 0
        for ngram in ngrams:
            if self.llm.tokenizer.encode(self.keywords)[0] in ngram:
                cnt += 1
        
        return cnt / len(ngrams)