from typing import TypedDict, Callable, Dict


class EStep(TypedDict):
    method: str
    walkers_per_dim: int
    nsteps: int
    nburnin: int
    sampler: str
    show_progress: bool


class MStep(TypedDict):
    minimize_method: str
    imputation_function: Callable[[int], int]  # Function type


class Convergence(TypedDict):
    criterion: str
    default: Dict[int, float]


class EMConfigType(TypedDict):
    max_steps: int  # Max steps until force exit.
    method: str  # DEFAULT or INVERTED method.
    verbose: bool
    e_step: EStep
    m_step: MStep
    convergence: Convergence
