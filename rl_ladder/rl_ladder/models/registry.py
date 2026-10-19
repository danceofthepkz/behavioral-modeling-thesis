"""Model registry: a model is a choice of variant for each of three slots.

    stimulus: none | S | S_cf | A        stimulus-specific value learning
    action:   none | V | gamma | wsls    reward-dependent, stimulus-independent
    choice:   none | ck                  reward-independent repetition

The bias b is always free unless a model fixes it. Models are declared in
config/models.yml; this module turns each entry into a ModelSpec, derives its
free parameters, and rejects combinations the kernel cannot express.
"""

from dataclasses import dataclass, field

import numpy as np

from . import params as P

SLOTS = ("stimulus", "action", "choice")
SLOT_PARAMS = {
    "stimulus": {"none": (), "S": ("alpha", "beta_q", "q0"),
                 "S_cf": ("alpha", "beta_q", "q0"), "A": ("alpha", "q0")},
    "action": {"none": (), "V": ("alpha_v", "beta_v"), "gamma": ("gamma",), "wsls": ("eps",)},
    "choice": {"none": (), "ck": ("alpha_c", "beta_c")},
}
BASE_CODE = {"none": 0, "S": 1, "S_cf": 1, "A": 2}
RESETTABLE = ("stimulus", "action", "choice", "wsls")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    stimulus: str
    action: str
    choice: str
    free: tuple
    fixed: dict = field(default_factory=dict)
    about: str = ""

    @property
    def base(self):
        return BASE_CODE[self.stimulus]

    @property
    def k(self):
        return len(self.free)

    @property
    def slug(self):
        return self.name.replace("+", "_")

    @property
    def free_specs(self):
        return [P.BY_NAME[n] for n in self.free]

    @property
    def depth(self):
        return sum(getattr(self, s) != "none" for s in SLOTS)

    def template(self):
        """Theta with every non-free parameter at its neutral or fixed value."""
        theta = P.neutral_theta()
        if self.stimulus == "S_cf":
            theta[P.KAPPA] = 1.0
        for name, value in self.fixed.items():
            theta[P.BY_NAME[name].index] = float(value)
        return theta

    def theta(self, values):
        """Full theta from a {name: value} dict of free parameters."""
        theta = self.template()
        for name in self.free:
            theta[P.BY_NAME[name].index] = float(values[name])
        return theta

    def to_dict(self):
        return {"name": self.name, "stimulus": self.stimulus, "action": self.action,
                "choice": self.choice, "free": list(self.free),
                "fixed": dict(sorted(self.fixed.items())), "k": self.k}


def make_spec(name, entry):
    slots = {s: str(entry.get(s, "none")) for s in SLOTS}
    for slot, variant in slots.items():
        if variant not in SLOT_PARAMS[slot]:
            raise ValueError(f"{name}: unknown {slot} variant {variant!r}; "
                             f"choose from {sorted(SLOT_PARAMS[slot])}")
    if slots["action"] == "gamma" and slots["stimulus"] != "A":
        raise ValueError(f"{name}: action 'gamma' modifies the A update and needs stimulus: A")
    fixed = {k: float(v) for k, v in (entry.get("fixed") or {}).items()}
    extra = list(entry.get("free") or [])
    for n in list(fixed) + extra:
        if n not in P.BY_NAME:
            raise ValueError(f"{name}: unknown parameter {n!r}")
    names = {"b"} | set(extra)
    for slot, variant in slots.items():
        names |= set(SLOT_PARAMS[slot][variant])
    names -= set(fixed)
    free = tuple(p.name for p in P.PARAMS if p.name in names)  # canonical order
    return ModelSpec(name=name, free=free, fixed=fixed, about=str(entry.get("about", "")), **slots)


def build_registry(models_yaml):
    models = {name: make_spec(name, entry) for name, entry in models_yaml["models"].items()}
    sets = {k: list(v) for k, v in (models_yaml.get("sets") or {}).items()}
    for set_name, members in sets.items():
        unknown = [m for m in members if m not in models]
        if unknown:
            raise ValueError(f"Model set {set_name!r} lists unknown models {unknown}")
    return models, sets


def resolve_models(arg, models, sets):
    """'core' -> set members; 'A0,A+R' -> list; 'S+C' -> [S+C]."""
    names = []
    for token in ([arg] if isinstance(arg, str) else list(arg)):
        for part in str(token).split(","):
            part = part.strip()
            if not part:
                continue
            if part in sets:
                names.extend(sets[part])
            elif part in models:
                names.append(part)
            else:
                raise KeyError(f"Unknown model or model set {part!r}. "
                               f"Models: {sorted(models)}; sets: {sorted(sets)}")
    return list(dict.fromkeys(names))


def parents(spec, models):
    """Registered models nested in `spec` by switching exactly one slot off.

    Used for warm starts: the parent optimum, with the new module at its
    neutral value, is a start whose NLL equals the parent's, so the child fit
    can never end worse than its parent.
    """
    out = []
    for other in models.values():
        if other.name == spec.name or other.stimulus != spec.stimulus or other.fixed != spec.fixed:
            continue
        diff = [s for s in SLOTS if getattr(other, s) != getattr(spec, s)]
        if len(diff) == 1 and getattr(other, diff[0]) == "none" and set(other.free) < set(spec.free):
            out.append(other.name)
    return sorted(out)


def reset_flags(session_reset):
    unknown = set(session_reset) - set(RESETTABLE)
    if unknown:
        raise ValueError(f"model.session_reset: unknown entries {sorted(unknown)}; "
                         f"choose from {RESETTABLE}")
    return np.array([int(s in session_reset) for s in RESETTABLE], dtype=np.int64)
