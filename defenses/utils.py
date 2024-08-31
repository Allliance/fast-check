
def get_defense(name, model, tokenizer):
    if name.lower() == "self-defense":
        from .self_defense import SelfDefense
        return SelfDefense(model, tokenizer)
    else:
        raise ValueError(f"Defense {name} not found")