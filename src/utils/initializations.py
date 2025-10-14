import numpy as np

def initialization_router(initialization_name:str, n_in: int, n_out: int, ) -> np.ndarray:
    match initialization_name:
        case "xavier":
            return np.sqrt(2. / (n_in + n_out))
        case "he":
            return np.sqrt(2. / n_in)
        case "le_cun":
            return np.sqrt(1. / n_in)
        case _:
            raise ValueError(f"Initilization function '{initialization_name}' not recognized.")
        



