
def get_defense(name, model, **kwargs):
    if name.lower() == "self-defense":
        from .self_defense import SelfDefense
        model = model.model
        tokenizer = model.tokenizer
        return SelfDefense(model, tokenizer)
    else:
        raise ValueError(f"Defense {name} not found")