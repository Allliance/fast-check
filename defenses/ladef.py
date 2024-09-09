from .base import BaseDefense

KEYWORDS = ["sorry", "I can't", "I can not", "I cannot", "I'm sorry", "I am sorry", "I can't do that", "I can not do that", "I cannot do that"]

class LADefense(BaseDefense):
    def __init__(self,
                 model,
                 keywords=KEYWORDS,
                 ):
        super().__init__("LADefense", model)
        assert model.use_lade, "Model must be loaded in lookahead mode to use this defense"

    def _is_jailbreak(self, prompt):
        response = self.get_response(prompt, include_prefix=False, max_new_tokens=self.needed_tokens)
        
        harm_prdiction_prompt = make_harm_prediction_prompt(response)
        self_evaluation = self.get_response(harm_prdiction_prompt, max_new_tokens=30).lower()
        
        assert self_evaluation.startswith("yes") or self_evaluation.startswith("no"), \
        f"Self-Defense model did not return a valid response. Got: {self_evaluation if self_evaluation else 'Empty response'}"
        
        return self_evaluation.startswith("yes")
    