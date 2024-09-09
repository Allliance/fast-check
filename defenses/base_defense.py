

class BaseDefense:
    def __init__(self, name, model):
        self.name = name
        self.model = model
    
    def get_response(self, prompt, include_prefix=True, max_new_tokens=1024):
        output_ids = self.model.tokenizer.encode(self.model.chat(prompt, max_new_tokens))
        
        if not include_prefix:
            prompt_length = len(self.tokenizer.encode(prompt))
            output_ids = output_ids[prompt_length:]
            
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()  

    def __str__(self) -> str:
        return self.name
    
    
    def is_jailbreak(self, prompts):
        if not isinstance(prompts, list):
            prompts = [prompts]
        assert all([isinstance(p, str) for p in prompts]), "Prompts must be a list of strings"
        
        return [self._is_jailbreak(p) for p in prompts]
        