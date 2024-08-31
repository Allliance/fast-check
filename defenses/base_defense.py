

class BaseDefense:
    def __init__(self, name, model, tokenizer):
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
    
    def get_full_response(self, prompt, include_prefix=True):
        output_ids = self.model.generate(
            self.tokenizer.encode(prompt, return_tensors="pt", max_length=1024).to(self.model.device),
            max_length=1024,
            top_p=0.9,
            do_sample=True,
            temperature=0.7
        )[0]
        if not include_prefix:
            prompt_length = len(self.tokenizer.encode(prompt))
            output_ids = output_ids[prompt_length:]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)
        
    
    def is_jailbreak(self, prompt):
        raise NotImplementedError