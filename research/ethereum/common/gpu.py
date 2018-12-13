import pyamgx
import numpy as np

def get_config():
    return pyamgx.Config().create_from_dict({
       "config_version": 2,
            "determinism_flag": 1,
            "exception_handling" : 1,
            "solver": {
                "monitor_residual": 1,
                "solver": "BICGSTAB",
                "convergence": "RELATIVE_INI_CORE",
                "preconditioner": {
                    "solver": "NOSOLVER"
            }
        }
    })

def solve_gpu(A_value, b_value, initial_x=None, finalize=True):
    A = pyamgx.Matrix().create(resource)
    b = pyamgx.Vector().create(resource)
    x = pyamgx.Vector().create(resource)

    solver = pyamgx.Solver().create(resource, config)

    if initial_x:
        solution = initial_x
    else:
        solution = np.zeros(A_value.shape[1], dtype=np.float64)

    A.upload_CSR(A_value)
    b.upload(b_value)
    x.upload(solution)

    solver.setup(A)
    solver.solve(b, x)

    x.download(solution)

    A.destroy()
    x.destroy()
    b.destroy()
    solver.destroy()
    resource.destroy()
    config.destroy()

    if finalize:
        pyamgx.finalize()

    return solution

pyamgx.initialize()
config = get_config()
resource = pyamgx.Resources().create_simple(config)
