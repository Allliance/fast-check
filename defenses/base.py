

class BaseDefense:
    def __init__(self, name, llm):
        self.name = name
        self.llm = llm
    
    def get_response(self, prompt, include_prefix=True, max_new_tokens=1024):
        tokenizer = self.llm.tokenizer
        output_ids = tokenizer.encode(self.llm.chat(prompt, max_new_tokens)[0])
        
        
        if not include_prefix:
            prompt_length = len(tokenizer.encode(prompt))
            output_ids = output_ids[prompt_length:]
        return tokenizer.decode(output_ids, skip_special_tokens=True).strip()  

    def __str__(self) -> str:
        return self.name
    
    
    def is_jailbreak(self, prompts):
        if not isinstance(prompts, list):
            prompts = [prompts]
        assert all([isinstance(p, str) for p in prompts]), "Prompts must be a list of strings"
        
        return [self._is_jailbreak(p) for p in prompts]
        