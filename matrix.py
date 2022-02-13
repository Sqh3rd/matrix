from __future__ import annotations
from math import ceil
import numpy as np
from copy import deepcopy

class Matrix:
    def __init__(self,
                 lines: int = 2,
                 columns: int = 2,
                 values: list = [[0, 0], [0, 0]]) -> None:
        if len(values) < 1:
            values = [[0]]
        elif len(values[0]) < 1:
            for val in values:
                val = [0]
        elif type(values) == np.ndarray:
            values = values.tolist()
        if lines < 1:
            lines = 1
        if columns < 1:
            columns = 1
        if values != [[0, 0], [0, 0]]:
            self.values = values
            self.lengths = []
            for line in self.values:
                self.lengths.append(len(line))
            max_len = sorted(self.lengths, reverse=True)[0]
            for line in self.values:
                while len(line) < max_len:
                    line.append(0)
        else:
            self.values = [[0 for a in range(columns)] for b in range(lines)]

    def lines(self) -> int:
        return len(self.values)
    
    def columns(self) -> int:
        return len(self.values[0])

    def add(self, x) -> Matrix:
        if not type(x) is Matrix:
            raise NotImplementedError(f"Addition with {type(x)} is not implemented!")
        if not (len(x.values) == len(self.values)
                and len(x.values[0]) == len(self.values[0])):
            raise ValueError(f"Addition with Matrix with an unequal amount of lines and columns is not possible!")
        return Matrix(
            values=[[x.values[i][c] + d for c, d in enumerate(self.values[i])]
                    for i in range(len(self.values))])

    def sub(self, first_val, second_val) -> Matrix:
        if type(first_val) is Matrix and type(second_val) is Matrix:
            first_val = first_val.values
            second_val = second_val.values
            if not len(first_val) == len(second_val) and len(
                    first_val[0]) == len(second_val[0]):
                raise ValueError(f"Subtraction with Matrix with an unequal amount of lines and columns is not possible!")
            return Matrix(values=[[
                first_val[i][c] - d for c, d in enumerate(second_val[i])
            ] for i in range(len(second_val))])
        raise NotImplementedError(f"Subtraction with {type(first_val) if type(first_val) != Matrix else type(second_val)} is not implemented!")

    def mul(self, first_value: int or float or Matrix, second_value: int
            or float or Matrix) -> Matrix:
        if (type(first_value) is int or type(first_value) is float) or (
                type(second_value) is int or type(second_value) is float):
            if type(first_value) is Matrix:
                return Matrix(values=[[val * second_value for val in innerval]
                                      for innerval in first_value.values])
            elif type(second_value) is Matrix:
                return Matrix(values=[[val * first_value for val in innerval]
                                      for innerval in second_value.values])
        elif type(first_value) is Matrix and type(second_value) is Matrix:
            if not first_value.columns() == second_value.lines():
                raise Exception(
                    "The first Matrix must have the same amount of columns as the amount of lines the second Matrix has!"
                )
            new_values = []
            for i in range(first_value.lines()):
                new_values.append([])
                for j in range(second_value.columns()):
                    new_values[i].append(
                        sum([
                            first_value.values[i][a] *
                            second_value.values[a][j]
                            for a in range(first_value.columns())
                        ]))
            return Matrix(values=new_values)
        raise NotImplementedError(f"Multiplication with {type(first_value) if type(first_value) != Matrix else type(second_value)} is not implemented!")

    def transpose(self) -> Matrix:
        return Matrix(values=[[self.values[i][j] for i in range(self.lines())]
                              for j in range(self.columns())])

    def edit(self, index: tuple, value: int or float) -> None:
        self.values[index[0]][index[1]] = value

    def edit(self, indices: list[tuple], values: list[int or float]) -> None:
        for i in range(len(indices)):
            self.values[indices[i][0]][indices[i][1]] = values[i]

    def insert_line(self, index: int) -> None:
        if index < 0:
            index = len(self.values) - index
        self.values.insert(index, [0 for i in range(self.columns())])

    def insert_column(self, index: int) -> None:
        if index < 0:
            index = len(self.values[0]) - index
        for i in range(self.lines()):
            self.values[i].insert(index, 0)

    def insert_outer_boundary(self) -> None:
        self.insert_line(0)
        self.insert_line(-1)
        self.insert_column(0)
        self.insert_column(-1)

    def crop_to_value(self, value) -> None:
        max_val = 0
        for val in self.values:
            for v in val:
                if v > max_val:
                    max_val = v
        ratio = value/max_val
        for a in range(self.lines()):
            for b in range(self.columns()):
                self.values[a][b] *= ratio

    def replace(self, old_number:float, new_number:float, return_matrix=False) -> None or Matrix:
        temp_vals = self.values
        for i in range(len(temp_vals)):
            for j in range(len(temp_vals[i])):
                if temp_vals[i][j] == old_number:
                    temp_vals[i][j] = new_number
        if return_matrix:
            return Matrix(values=temp_vals)
        self.values=temp_vals

    def replace_multiple(self, old_numbers:list[float], new_number:float, return_matrix=False) -> None or Matrix:
        temp_vals = self.values
        for i in range(len(temp_vals)):
            for j in range(len(temp_vals[i])):
                if temp_vals[i][j] in old_numbers:
                    temp_vals[i][j] = new_number
        if return_matrix:
            return Matrix(values=temp_vals)
        self.values=temp_vals
    
    def replace_between(self, inclusive_minimum:float, inclusive_maximum:float, new_number:float, return_matrix=False) -> None or Matrix:
        temp_vals = self.values
        for i in range(len(self.values)):
            for j in range(len(self.values[i])):
                if inclusive_minimum <= temp_vals[i][j] <= inclusive_maximum:
                    temp_vals[i][j] = new_number
        if return_matrix:
            return Matrix(values=temp_vals)
        self.values = temp_vals

    def count_values(self) -> list[list]:
        try:
            end_list = [[0,0]]
            for value in self.values:
                for val in value:
                    is_in_list = False
                    for e in end_list:
                        if val == e[1]:
                            e[0] += 1
                            is_in_list = True
                    if not is_in_list:
                        end_list.append([1,val])
            return end_list
        except Exception as e:
            print(e)
            raise Exception()

    def __repr__(self) -> str:
        return f"{len(self.values)}x{len(self.values[0])}-Matrix"

    def __str__(self) -> str:
        strings = [["|".join([str(b) for b in a])] for a in self.values]
        return "\n".join([
            str(string).replace("[", "").replace("]", "").replace("'", "")
            for string in strings
        ])

    def __add__(self, x) -> Matrix:
        return self.add(x)

    def __radd__(self, x) -> Matrix:
        return self.add(x)

    def __sub__(self, x) -> Matrix:
        return self.sub(self, x)

    def __rsub__(self, x) -> Matrix:
        return self.sub(x, self)

    def __mul__(self, x) -> Matrix:
        return self.mul(self, x)

    def __rmul__(self, x) -> Matrix:
        return self.mul(x, self)

    def __eq__(self, x) -> bool:
        if not type(x) is Matrix:
            return False
        if not len(x.values) == len(self.values) and len(x.values[0]) == len(
                self.values[0]):
            return False
        for i, line in enumerate(self.values):
            for j, val in enumerate(line):
                if not val == x.values[i][j]:
                    return False
        return True

class Flags:
    AVERAGE_POOLING = 0
    MAX_POOLING = 1

def kernel_multiplicate(first_matrix: Matrix,
                        second_matrix: Matrix,
                        stride_length: int = 1,
                        crop_to_val: int = 0,
                        get_average: bool = False,
                        get_difference: bool = False,
                        keep_size: bool=False) -> Matrix:
    if first_matrix.columns() > second_matrix.columns() and first_matrix.lines() > second_matrix.lines():
        greater_value = deepcopy(first_matrix)
        smaller_value = deepcopy(second_matrix)
    elif first_matrix.columns() < second_matrix.columns() and first_matrix.lines() < second_matrix.lines():
        greater_value = deepcopy(second_matrix)
        smaller_value = deepcopy(first_matrix)
    else:
        raise Exception("One Matrix has to be smaller than the other!")
    if ((greater_value.lines() - smaller_value.lines()) % stride_length != 0 or
        (greater_value.columns() - smaller_value.columns()) % stride_length != 0):
        raise ValueError("Amount of Lines and Columns have to be dividable by the stride length!")
    if keep_size:
        greater_value.insert_outer_boundary()
    result_matrix = Matrix(
        lines=int((greater_value.lines() - smaller_value.lines()) / stride_length) + 1,
        columns=int((greater_value.columns() - smaller_value.columns()) / stride_length) + 1
    )
    temp_sum = 0
    if crop_to_val != 0:
        max_sum = 0
    values_for_new_matrix = []
    inc = 0
    for lin_stride in range(0, greater_value.lines() - smaller_value.lines(),
                            stride_length):
        values_for_new_matrix.append([])
        for col_stride in range(0,
                                greater_value.columns() - smaller_value.columns(),
                                stride_length):
            if not get_difference:
                temp_sum = sum([
                    greater_value.values[lin + lin_stride][col + col_stride] *
                    smaller_value.values[lin][col]
                    for col in range(0, smaller_value.columns(), 1)
                    for lin in range(0, smaller_value.lines(), 1)
                ])
            else:
                if smaller_value.columns() % 2 == 0 or smaller_value.lines() % 2 == 0:
                    raise Exception("The smaller Matrix has to have an uneven number of lines and columns!")
                middle_col = int(smaller_value.columns() / 2 - (1 if smaller_value.columns() / 2 % 2 == 0 else 0.5))
                middle_lin = int(smaller_value.lines() / 2 - (1 if smaller_value.columns() / 2 % 2 == 0 else 0.5))
                for lin in range(smaller_value.lines()):
                    for col in range(smaller_value.columns()):
                        if col == middle_col and lin == middle_lin:
                            continue
                        temp_sum += abs(
                            greater_value.values[lin + lin_stride][
                                col + col_stride] *
                            smaller_value.values[lin][col] -
                            greater_value.values[middle_lin + lin_stride][
                                middle_col + col_stride] *
                            smaller_value.values[lin][col])
            if get_average:
                temp_sum = temp_sum / (smaller_value.lines() *
                                       smaller_value.columns())
            values_for_new_matrix[inc].append(temp_sum if temp_sum > 0 else 0)
            if crop_to_val != 0:
                if temp_sum > max_sum:
                    max_sum = temp_sum
        inc += 1
    if crop_to_val == 0:
        result_matrix.edit(indices=[
            (a, b) for a in range(len(values_for_new_matrix))
            for b in range(len(values_for_new_matrix[a]))
        ],
                           values=[
                               values_for_new_matrix[a][b]
                               for a in range(len(values_for_new_matrix))
                               for b in range(len(values_for_new_matrix[a]))
                           ])
    else:
        scale_factor = crop_to_val / (max_sum if max_sum > 0 else 1)
        result_matrix.edit(
            indices=[(a, b) for a in range(len(values_for_new_matrix))
                     for b in range(len(values_for_new_matrix[a]))],
            values=[
                ceil(values_for_new_matrix[a][b] * scale_factor) if
                ceil(values_for_new_matrix[a][b] * scale_factor) < 255 else 255
                for a in range(len(values_for_new_matrix))
                for b in range(len(values_for_new_matrix[a]))
            ])
    return result_matrix

def pooling(inp_matrix:Matrix, pool_kernel_size:tuple, avg_or_max:int, iterations=1) -> Matrix:
    matrix = inp_matrix
    for i in range(iterations):
        end_arr = []
        if matrix.lines() % pool_kernel_size[0] != 0 or matrix.columns() % pool_kernel_size[1] != 0:
            raise Exception("Lines and Columns of Input Matrix have to be dividable by the Pool Kernel Size!")
        for i in range(0, matrix.lines(), pool_kernel_size[0]):
            end_arr.append([])
            for j in range(0, matrix.columns(), pool_kernel_size[1]):
                current_rect = [matrix.values[k][h] for k in range(i, i + pool_kernel_size[0]) for h in range(j, j + pool_kernel_size[1])]
                if avg_or_max == 0:
                    result = sum(current_rect)/(pool_kernel_size[0]*pool_kernel_size[1])
                else:
                    result = max(current_rect)
                end_arr[-1].append(result)
        matrix = Matrix(values=end_arr)
    return matrix

if __name__ == '__main__':
    a = Matrix(values=[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    b = Matrix(lines=5, columns=5)
    print(a)
    print(pooling(a, (2,2), Flags.AVERAGE_POOLING))
    print(pooling(a, (2,2), Flags.MAX_POOLING))