from typing import TypedDict, Dict


class EMConfigType(TypedDict):
    max_steps: int
    method: str
    convergence_ths: int
    sampling_params: Dict[
        str, dict
    ]  # Assuming all values in sampling_params are dictionaries
