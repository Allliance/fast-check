import torch

from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
)

def load_model_and_tokenizer(
    model_name,
    use_lade=False,
    debug=False,
    **kwargs,
    ):

    device = 'cuda:1'

    if model_name == 'llama':
        model_name = "meta-llama/Llama-2-7b-chat-hf"
    else:
        raise ValueError(f"Invalid Model Name: {model_name}")
    # model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    model_kwargs = {
        'pretrained_model_name_or_path': model_name,
        'device_map': 'auto',
        # 'torch_dtype': torch.float16,
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

    if use_lade:
        import os, lade
        os.environ['USE_LADE'] = os.environ['LOAD_LADE'] = str(1)
        torch_device = 'cuda:1'
        
        lade.augment_all()
        #For a 7B model, set LEVEL=5, WINDOW_SIZE=7, GUESS_SET_SIZE=7
        lade.config_lade(LEVEL=5, WINDOW_SIZE=7, GUESS_SET_SIZE=7, DEBUG=1 ,POOL_FROM_PROMPT=True)
        # model_kwargs['attn_implementation'] = "flash_attention_2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        **model_kwargs
    )
    
    if not debug:
        model.half()
    model.tokenizer = tokenizer
    
    return model, tokenizer