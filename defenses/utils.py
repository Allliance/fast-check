
def get_defense(name, llm, **kwargs):
    if name.lower() == "self-defense":
        from .self_defense import SelfDefense
        return SelfDefense(llm, **kwargs)
    elif name.lower() == "ladef":
        from .ladef import LADefense
        return LADefense(llm, **kwargs)
    else:
        raise ValueError(f"Defense {name} not found")