import os
import gc
import torch
from fastchat.model import get_conversation_template
from fastdef.models_config import MODELS_CONFIG
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def load_model_and_tokenizer(
        model_name,
        debug,
        use_lade,
        lade_level=4,
        lade_window_size=10,
        lade_guess_set_size=-1,
        lade_use_prompt_pool=False,
        lade_debug=False,
        **kwargs
        ):

    print("Loading Model:", model_name, "...")
    
    model_hf_name = MODELS_CONFIG[model_name]['model_name']
    
    tokenizer = AutoTokenizer.from_pretrained(model_hf_name,
            trust_remote_code=True,
            use_fast=False)
    
    model_kwargs = {
        'pretrained_model_name_or_path': model_hf_name,
        'low_cpu_mem_usage': True,
        'trust_remote_code': True,
        'use_cache': True,
    }


    if debug:
        attn_implementation = "eager"

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs['attn_implementation'] = attn_implementation
        model_kwargs['quantization_config'] = bnb_config
    else:
        model_kwargs.update({
            'device_map': 'auto',
        })

    if use_lade:
        from fastdef.libs import lade
        os.environ['USE_LADE'] = os.environ['LOAD_LADE'] = str(1)
        
        print("Using Lookahead decoding")
        print("Lade parameters: LEVEL={}, WINDOW_SIZE={}, GUESS_SET_SIZE={}".format(
            lade_level, lade_window_size, lade_guess_set_size
        ))
        lade.augment_all()
        #For a 7B model, set LEVEL=5, WINDOW_SIZE=7, GUESS_SET_SIZE=7
        lade.config_lade(LEVEL=lade_level,
                         WINDOW_SIZE=lade_window_size,
                         GUESS_SET_SIZE=lade_guess_set_size,
                         DEBUG=lade_debug,
                         POOL_FROM_PROMPT=lade_use_prompt_pool)
        # model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        **model_kwargs
    )
    
    torch.cuda.empty_cache()
    gc.collect()
    
    if not debug:
        model.half()
        
    model.tokenizer = tokenizer
    
    print("Model Loaded Successfully!")
    
    return model, tokenizer

class ChatModel:

    def __init__(
        self,
        model_name="llama2",
        debug=False,
        use_lade=False,
        **kwargs
    ):
        self.name = model_name
        self.use_lade = use_lade
        
        # Language model
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_name=model_name,
            debug=debug,
            use_lade=use_lade,
            **kwargs
        )
        
        if debug:
            print("Running in DEBUG mode - model is loaded in 4bits")
        else:
            print("Running in DEPLOY mode - model is loaded in 16bits")
            
        self.tokenizer.padding_side = 'left'
        if model_name == 'llama2':
            self.tokenizer.pad_token = self.tokenizer.unk_token
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.conv_template = get_conversation_template(
            MODELS_CONFIG[model_name]['conversation_template']
        )
        if self.conv_template.name == 'llama-2':
            self.conv_template.sep2 = self.conv_template.sep2.strip()
        

    def chat(self,
             prompts,
             max_new_tokens=1024,
             **kwargs):

        torch.cuda.empty_cache()

        if not isinstance(prompts, list):
            prompts = [prompts]

        full_prompts = []
        
        for prompt in prompts:
            self.conv_template.append_message(self.conv_template.roles[0], prompt)
            self.conv_template.append_message(self.conv_template.roles[1], "")
            prompt = self.conv_template.get_prompt()
            
            encoding = self.model.tokenizer(prompt)
            full_prompt = self.model.tokenizer.decode(
                encoding.input_ids
            ).replace('<s>','').replace('</s>','')
            
            full_prompts.append(full_prompt)
            
            self.conv_template.messages = []

        batch_outputs = self.__call__(full_prompts, max_new_tokens, **kwargs)

        return batch_outputs

    @torch.no_grad()
    def __call__(self,
                 batch,
                 max_new_tokens=1024,
                 return_ngrams=False,
                 top_p=0.9,
                 do_sample=True,
                 temperature=0.6,
                 **kwargs):

        if not isinstance(batch, list):
            batch = [batch]

        # Pass current batch through the tokenizer
        batch_inputs = self.tokenizer(
            batch,
            padding=True,
            truncation=False, 
            return_tensors='pt'
        )
        
        batch_input_ids = batch_inputs['input_ids'].to(self.model.device)
        batch_attention_mask = batch_inputs['attention_mask'].to(self.model.device)

        # Forward pass through the LLM
        try:
            outputs = self.model.generate(
                batch_input_ids, 
                attention_mask=batch_attention_mask, 
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=return_ngrams,
                top_p=top_p,
                do_sample=do_sample,
                temperature=temperature,
                **kwargs
            )
            if return_ngrams:
                assert len(batch_input_ids) == 1, "Only works for single batch"
                
                final_response = outputs[0][0][len(batch_input_ids[0]):]
                return (final_response, outputs[1])
        except RuntimeError as e:
            raise e
            return []

        # Decode the outputs produced by the LLM
        batch_outputs = self.tokenizer.batch_decode(
            outputs, 
            skip_special_tokens=True
        )
        
        gen_start_idx = [
            len(self.tokenizer.decode(batch_input_ids[i], skip_special_tokens=True)) 
            for i in range(len(batch_input_ids))
        ]
        batch_outputs = [
            output[gen_start_idx[i]:] for i, output in enumerate(batch_outputs)
        ]
        
        torch.cuda.empty_cache()

        return batch_outputs