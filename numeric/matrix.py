from __future__ import annotations
from typing import List

class Matrix:
    """
    Creates a :class:`Matrix` with given elements and shape.

    Args:
        elements: a list containing the elements of the matrix.
        shape: (n_rows, n_cols) tuple formed by the number of rows and cols of the matrix.
    
    Returns:
        The newly created :class:`Matrix`.
    
    Raises:
        ValueError: 
            if n_rows or n_cols is less than 1
            or if n_rows*n_cols doesn't match the number of elements.

    .. automethod:: __add__
    .. automethod:: __sub__
    .. automethod:: __mul__
    """

    def __init__(self, elements: list[complex], shape: tuple[int, int]) -> Matrix:
        n_rows, n_cols = shape
        if n_rows <= 0 or n_cols <= 0:
            raise ValueError('Invalid shape {}'.format(shape))
        
        if len(elements) != n_rows * n_cols:
            raise ValueError('Inconsistent shape {}: {} * {} = {} but the number of elements is {}'.format(
                shape,
                n_rows,
                n_cols,
                n_rows * n_cols,
                len(elements)
            ))

        self._elements = elements
        self._shape = shape
        pass

    @classmethod
    def fromRows(cls, rows: list[list[complex]]) -> Matrix:
        """
        Creates a :class:`Matrix` from a list of rows.

        Args:
            rows: a list rows. Each row is a list of :class:`complex` 
                with the same number of elements.
        
        Returns:
            The newly created :class:`Matrix`
        
        Raises:
            ValueError: if number of rows or cols is less than 1
                or if the number of elements is not the same in each row.
        """
        n_rows = len(rows)
        if n_rows < 1:
            raise ValueError('Number of rows must be at least 1')

        n_cols = len(rows[0])
        if n_cols < 1:
            raise ValueError('Number of cols must be at least 1')

        elements = []
        for row in rows:
            if len(row) != n_cols:
                raise ValueError('Number of elements must be the same in each row')

            elements += row

        shape = (n_rows, n_cols)
        return cls(elements, shape)
    
    @classmethod
    def makeVector(cls, elements: list[complex]) -> Matrix:
        """
        Creates a :class:`Matrix` that is a column vector of given elements.

        Args:
            elements: a list containg the elements of the vector.
        
        Returns:
            The newly created :class:`Matrix` representig the column vector.
        
        Raises:
            ValueError: if the number of elements is 0.
        """
        if len(elements) == 0:
            raise ValueError('Number of elements must be greater than 0')
        return Matrix(elements, (len(elements), 1))

    def elements(self) -> list[complex]:
        """
        Returns:
            A list containing all the elements of the matrix.
        """
        return self._elements

    def shape(self) -> tuple[int, int]:
        """
        Returns:
            A tuple like (n_rows, n_cols) representing the shape of the matrix.
        """
        return self._shape

    def at(self, row: int, col: int) -> complex:
        """
        Returns the element of position (row, col).

        Args:
            row: index of the row.
            col: index of the column.
        
        Returns:
            The element of position (row, col).
        
        Raises:
            IndexError: if row or col is not a valid index.
        """
        n_rows, n_cols = self._shape
        if row < 0 or row >= n_rows:
            raise IndexError('Row index {} out of bound'.format(row))
        if col < 0 or col >= n_cols:
            raise IndexError('Col index {} out of bound'.format(col))

        index = row * n_cols + col
        return self._elements[index]

    def rows(self) -> list[list[complex]]:
        """
        Returns:
            The list of rows that makes up the matrix.
        """
        n_rows, n_cols = self._shape
        rows = []
        for row in range(n_rows):
            si = row * n_cols
            ei = (row + 1) * n_cols
            rows.append(self._elements[si:ei])
        return rows
    
    def cols(self) -> list[list[complex]]:
        """
        Returns:
            The list of columns that makes up the matrix.
        """
        n_rows, n_cols = self._shape
        cols = []
        for col in range(n_cols):
            cols.append(
                [self._elements[row * n_cols + col] for row in range(n_rows)]
            )
        return cols

    def T(self) -> Matrix:
        """
        Returns:
            The transpose of the matrix.
        """
        return Matrix.fromRows(self.cols())
    
    def H(self) -> Matrix:
        """
        Returns:
            The conjugate transpose of the matrix.
        """
        M = Matrix.fromRows(self.cols())
        for i in range(len(M._elements)):
            M._elements[i] = complex(M._elements[i]).conjugate()
        return M

    def scaled(self, scale: complex) -> Matrix:
        """
        Args:
            scale: A complex scalar.
        
        Returns:
            A new :class:`Matrix` whose elements are all the elements 
            of the original matrix multiplied by a scalar.
        """
        elements = [scale * e for e in self._elements]
        shape = self._shape
        return Matrix(elements, shape)


    def __add__(self, other: Matrix) -> Matrix:
        """
        Defines the behaviour of the '+' operator .

        Returns:
            A new :class:`Matrix` wich is the sum of self and other.
        
        Raises:
            ValueError: if the shapes of self and other are not the same.
        """
        if self._shape != other._shape:
            raise ValueError('Incompatible shapes {} and {}: they must be the same'.format(
                self._shape,
                other._shape
            ))
        elements = [
            self._elements[i] + other._elements[i]
            for i in range(len(self._elements))
        ]        
        return Matrix(elements, self._shape)
    
    def __sub__(self, other: Matrix) -> Matrix:
        if self._shape != other._shape:
            raise ValueError('Incompatible shapes {} and {}: they must be the same'.format(
                self._shape,
                other._shape
            ))
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
        if n_cols_s != n_rows_o:
            raise ValueError((
                'Incompatible shapes {} and {}: the number of cols '
                'of the first matrix must match the number of rows '
                'of the the second matrix'
            ).format(
                self._shape,
                other._shape
            ))

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