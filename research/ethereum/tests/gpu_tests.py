import unittest
import scipy
import scipy.sparse
import scipy.sparse.linalg
from common.gpu import solve_gpu
import numpy as np

class GPUTestCase(unittest.TestCase):
  def test_solve_gpu(self):
    A = np.matrix([[1, 2], [2, 4]], dtype=np.float64)
    A = scipy.sparse.csr_matrix(A)
    b = np.array([1, 2], dtype=np.float64)
    cpu_x, _ = scipy.sparse.linalg.bicgstab(A, b, tol=1e-3)
    gpu_x = solve_gpu(A, b)
    assert np.sum((cpu_x - gpu_x) ** 2) < 0.001    