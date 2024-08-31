from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model_and_tokenizer(
    model_name,
    use_lade=True
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
    model.half()
    model.tokenizer = tokenizer
    
    return model, tokenizer