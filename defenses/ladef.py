from .base import BaseDefense
import numpy as np

KEYWORDS = ["I'm sorry", "Sorry", "sorry", "I am sorry", "I apologize", "As an", "As an AI", "I'm an", "I'm just", "As a language model", "As an Assistant", "I cannot", "I can't", "I do not", "cannot", "is illegal and unethical", "I'm really sorry", "unethical", "not ethical", "illegal", "not legal", "My apologies", "but I cannot", "I'm not able to provide", "I am really sorry", "I cannot create", "I cannot support", "I can't provide", "I can't assist", "I am unable", "I must", "Unfortunately", "unfortunately", "is not capable of", "As a responsible"]

def check_ngram(ngram, keywords, tk):
    for keyword in keywords:
        enc_kw = tk.encode(keyword)[1:]
        if tk.decode(ngram).find(keyword) != -1:
            return True
        for i in range(len(ngram) - len(enc_kw) + 1):
            if ngram[i:i+len(enc_kw)] == enc_kw:
                return True
    return False


class LADefense(BaseDefense):
    def __init__(self,
                 model,
                 max_new_tokens=40,
                 keywords=KEYWORDS,
                 **kwargs,
                 ):
        super().__init__("LADefense", model)
        assert model.use_lade, "Model must be loaded in lookahead mode to use this defense"
        self.keywords = [k for k in keywords if len(model.tokenizer.encode(k)) <= 5]
        self.max_new_tokens = max_new_tokens

    def _is_jailbreak(self, prompt):
        response, ngrams = self.llm.chat([prompt],
                        temperature=1.5,
                        top_p=1,
                        max_new_tokens=40,
                        return_ngrams=True)
        
        is_malicious = False
        for ngram in ngrams:
            if check_ngram(ngram, self.keywords, self.llm.tokenizer):
                is_malicious = True
                # print("Found malicious ngram:\n", self.llm.tokenizer.decode(ngram))
                break
        # print(is_malicious)
        return is_malicious