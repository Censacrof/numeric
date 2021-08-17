from __future__ import annotations
from typing import List

class Matrix:
    def __init__(self, elements: list[complex], shape: tuple[int, int]):
        n_rows, n_cols = shape
        assert(n_rows > 0)
        assert(n_cols > 0)
        assert(len(elements) == n_rows * n_cols)

        self._elements = elements
        self._shape = shape
        pass

    @classmethod
    def fromRows(cls, rows: list[list[complex]]) -> Matrix:
        n_rows = len(rows)
        assert(n_rows >= 1)

        n_cols = len(rows[0])
        assert(n_cols >= 1)

        elements = []
        for row in rows:
            assert(len(row) == n_cols)
            elements += row

        shape = (n_rows, n_cols)
        return cls(elements, shape)
    
    @classmethod
    def makeVector(cls, elements: list[complex]) -> Matrix:
        assert(len(elements) > 0)
        return Matrix(elements, (len(elements), 1))
    

    def at(self, row, col) -> complex:
        n_rows, n_cols = self._shape
        assert(row >= 0 and row < n_rows)
        assert(col >= 0 and col < n_cols)

        index = row * n_cols + col
        return self._elements[index]

    def rows(self) -> list[list[complex]]:
        n_rows, n_cols = self._shape
        rows = []
        for row in range(n_rows):
            si = row * n_cols
            ei = (row + 1) * n_cols
            rows.append(self._elements[si:ei])
        return rows
    
    def cols(self) -> list[list[complex]]:
        n_rows, n_cols = self._shape
        cols = []
        for col in range(n_cols):
            cols.append(
                [self._elements[row * n_cols + col] for row in range(n_rows)]
            )
        return cols

    def T(self) -> Matrix:
        return Matrix.fromRows(self.cols())
    
    def H(self) -> Matrix:
        M = Matrix.fromRows(self.cols())
        for i in range(len(M._elements)):
            M._elements[i] = complex(M._elements[i]).conjugate()
        return M

    def scaled(self, scale: complex) -> Matrix:
        elements = [scale * e for e in self._elements]
        shape = self._shape
        return Matrix(elements, shape)


    def __add__(self, other: Matrix) -> Matrix:
        assert(self._shape == other._shape)
        elements = [
            self._elements[i] + other._elements[i]
            for i in range(len(self._elements))
        ]        
        return Matrix(elements, self._shape)
    
    def __sub__(self, other: Matrix) -> Matrix:
        assert(self._shape == other._shape)
        elements = [
            self._elements[i] - other._elements[i]
            for i in range(len(self._elements))
        ]        
        return Matrix(elements, self._shape)
    
    def __pos__(self) -> Matrix:
        return self

    def __neg__(self) -> Matrix:
        elements = [
            -self._elements[i]
            for i in range(len(self._elements))
        ]        
        return Matrix(elements, self._shape)

    def __mul__(self, other: Matrix) -> Matrix:
        n_rows_s, n_cols_s = self._shape
        n_rows_o, n_cols_o = other._shape
        assert(n_cols_s == n_rows_o)

        n_rows = n_rows_s
        n_cols = n_cols_o
        elements = []
        for row in range(n_rows_s):
            for col in range(n_cols_o):
                p = 0
                for off in range(n_cols_s):
                    p += self._elements[row * n_cols_s + off] \
                        * other._elements[off * n_cols_o + col]
                elements.append(p)
        
        shape = (n_rows, n_cols)
        return Matrix(elements, shape)

    def __str__(self) -> str:
        n_rows, n_cols = self._shape
        s = ''
        for row in range(n_rows):
            for col in range(n_cols):
                s += '{}\t\t'.format(self._elements[row * n_cols + col])
            s += '\n'
        return s