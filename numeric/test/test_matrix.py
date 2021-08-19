from numeric.matrix import Matrix
import unittest

class TestMatrix(unittest.TestCase):
    def test___init__(self):
        with self.assertRaises(ValueError):
            Matrix([1, 2, 3, 4, 5, 6], (-2, -3))        
        with self.assertRaises(ValueError):
            Matrix([1, 2, 3, 4, 5], (2, 3))

    def test_fromRows(self):
        with self.assertRaises(ValueError):
            Matrix.fromRows([[1, 2, 3, 4], [4, 5, 6], [7, 8, 9], [10, 11, 12]])        
        with self.assertRaises(ValueError):
            Matrix.fromRows([[]])        
        with self.assertRaises(ValueError):
            Matrix.fromRows([])
        A = Matrix.fromRows([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        self.assertEqual(A._shape, (4, 3))
        self.assertEqual(A._elements, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    
    def test_makeVector(self):
        elements = [1, 2, 3]
        x = Matrix.makeVector(elements)
        self.assertEqual(x._shape, (3, 1))
        self.assertEqual(x._elements, elements)
    
    def test_diag(self):
        with self.assertRaises(ValueError):
            Matrix.diag([])
        A = Matrix.diag([1, 2, 3])
        self.assertEqual(A._elements, [1, 0, 0, 0, 2, 0, 0, 0, 3])
        self.assertEqual(A._shape, (3, 3))
    
    def test_identity(self):
        with self.assertRaises(ValueError):
            Matrix.identity(0)
        with self.assertRaises(ValueError):
            Matrix.identity(-1)
        I = Matrix.identity(3)
        self.assertEqual(I._elements, [1, 0, 0, 0, 1, 0, 0, 0, 1])
    
    def test_get_at(self):
        A = Matrix.fromRows([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        with self.assertRaises(IndexError):
            A.get_at(-1, 2)
        with self.assertRaises(IndexError):
            A.get_at(4, 1)        
        with self.assertRaises(IndexError):
            A.get_at(1, -2)
        with self.assertRaises(IndexError):
            A.get_at(4, 3)
        self.assertEqual(A.get_at(1, 2), 6)
    
    def test_set_at(self):
        A = Matrix.fromRows([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        with self.assertRaises(IndexError):
            A.set_at(-1, 2, 42)
        with self.assertRaises(IndexError):
            A.set_at(4, 1, 42)        
        with self.assertRaises(IndexError):
            A.set_at(1, -2, 42)
        with self.assertRaises(IndexError):
            A.set_at(4, 3, 42)
        A.set_at(1, 2, 0)
        self.assertEqual(A.get_at(1, 2), 0)

    def test___getitem__(self):
        A = Matrix.fromRows([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        self.assertEqual(A[1, 2], A.get_at(1, 2))
    
    def test___setitem__(self):
        A = Matrix.fromRows([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        A[1, 2] = 0
        self.assertEqual(A.get_at(1, 2), 0)

    def test_rows(self):
        rows = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        A = Matrix.fromRows(rows)
        self.assertEqual(A.rows(), rows)

    def test_cols(self):
        A = Matrix.fromRows([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        self.assertEqual(A.cols(), [[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]])

    def test_T(self):
        rows = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        A = Matrix.fromRows(rows)
        self.assertEqual(A.T().cols(), rows)

    def test_H(self):
        J = complex(0, 1)
        rows = [[1, 1+J], [0, 2]]
        A = Matrix.fromRows(rows)
        self.assertEqual(A.H().rows(), [[1, 0], [1-J, 2]])
    
    def test_scaled(self):
        J = complex(0, 1)
        elements = [1, 2, 3, 4, 5, 6]
        A = Matrix(elements, (2, 3))
        self.assertEqual(A.scaled(-2*J)._elements, [e * (-2*J) for e in elements])

    def test___add__(self):
        elements_a = [1, 2, 3, 4, 5, 6]
        elements_b = [2, 4, 6, 8, 10, 12]
        A = Matrix(elements_a, (3, 2))
        B = Matrix(elements_b, (2, 3))
        with self.assertRaises(ValueError):
            A + B
        B = Matrix(elements_b, (3, 2))
        C = A + B
        self.assertEqual(C._elements, [
            elements_a[i] + elements_b[i]
            for i in range(len(elements_a))
        ])

    def test___sub__(self):
        elements_a = [1, 2, 3, 4, 5, 6]
        elements_b = [2, 4, 6, 8, 10, 12]
        A = Matrix(elements_a, (3, 2))
        B = Matrix(elements_b, (2, 3))
        with self.assertRaises(ValueError):
            A - B
        B = Matrix(elements_b, (3, 2))
        C = A - B
        self.assertEqual(C._elements, [
            elements_a[i] - elements_b[i]
            for i in range(len(elements_a))
        ])
    
    def test___mul__(self):
        elements_a = [1, 2, 3, 4, 5, 6]
        elements_b = [2, 4, 6, 8, 10, 12]
        A = Matrix(elements_a, (3, 2))
        B = Matrix(elements_b, (3, 2))
        with self.assertRaises(ValueError):
            A * B
        B = Matrix(elements_b, (2, 3))
        C = A * B
        self.assertEqual(C.rows(), [[18, 24, 30], [38, 52, 66], [58, 80, 102]])


if __name__ == '__main__':
    unittest.main()