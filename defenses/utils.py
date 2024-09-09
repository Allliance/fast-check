
def get_defense(name, llm, **kwargs):
    if name.lower() == "self-defense":
        from .self_defense import SelfDefense
        return SelfDefense(llm)
    else:
        raise ValueError(f"Defense {name} not found")