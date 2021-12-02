from __future__ import annotations


def kernel_multiplicate(first_value: Matrix,
                        second_value: Matrix,
                        stride_length: int = 1) -> Matrix:
    if first_value.columns > second_value.columns and first_value.lines > second_value.lines:
        greater_value = first_value
        smaller_value = second_value
    elif first_value.columns < second_value.columns and first_value.lines < first_value.lines:
        greater_value = second_value
        smaller_value = first_value
    else:
        raise ValueError
    if ((greater_value.lines - smaller_value.lines) % stride_length != 0 or
        (greater_value.columns - smaller_value.columns) % stride_length != 0):
        raise ValueError
    result_matrix = Matrix(
        lines=1 + int(
            (greater_value.lines - smaller_value.lines) / stride_length),
        columns=1 + int((greater_value.columns - smaller_value.columns)),
    )
    temp_sum = 0
    values_for_new_matrix = []
    for lin_stride in range(greater_value.lines - smaller_value.lines + 1):
        values_for_new_matrix.append([])
        for col_stride in range(greater_value.columns - smaller_value.columns +
                                1):
            temp_sum = sum([
                greater_value.values[lin + lin_stride * stride_length][
                    col + col_stride * stride_length] *
                smaller_value.values[lin][col]
                for col in range(smaller_value.columns)
                for lin in range(smaller_value.lines)
            ])
            values_for_new_matrix[lin_stride].append(temp_sum)
    result_matrix.edit(indices=[(a, b)
                                for a in range(len(values_for_new_matrix))
                                for b in range(len(values_for_new_matrix[a]))],
                       values=[
                           values_for_new_matrix[a][b]
                           for a in range(len(values_for_new_matrix))
                           for b in range(len(values_for_new_matrix[a]))
                       ])
    return result_matrix


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
        if lines < 1:
            lines = 1
        if columns < 1:
            columns = 1
        if values != [[0, 0], [0, 0]]:
            self.lines = len(values)
            self.values = values
            self.lengths = []
            for line in self.values:
                self.lengths.append(len(line))
            max_len = sorted(self.lengths, reverse=True)[0]
            for line in self.values:
                while len(line) < max_len:
                    line.append(0)
            self.columns = max_len
        else:
            self.lines = lines
            self.columns = columns
            self.values = [[0 for a in range(columns)] for b in range(lines)]

    def add(self, x) -> Matrix:
        if not type(x) is Matrix:
            raise NotImplementedError
        if not (len(x.values) == len(self.values)
                and len(x.values[0]) == len(self.values[0])):
            raise NotImplementedError
        return Matrix(
            values=[[x.values[i][c] + d for c, d in enumerate(self.values[i])]
                    for i in range(len(self.values))])

    def sub(self, first_val, second_val) -> Matrix:
        if type(first_val) is Matrix and type(second_val) is Matrix:
            first_val = first_val.values
            second_val = second_val.values
            if not len(first_val) == len(second_val) and len(
                    first_val[0]) == len(second_val[0]):
                raise NotImplementedError
            return Matrix(values=[[
                first_val[i][c] - d for c, d in enumerate(second_val[i])
            ] for i in range(len(second_val))])
        raise NotImplementedError

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
            if not first_value.columns == second_value.lines:
                raise Exception(
                    "The first Matrix must have the same amount of columns as the amount of lines the second Matrix has!"
                )
            new_values = []
            for i in range(first_value.lines):
                new_values.append([])
                for j in range(second_value.columns):
                    new_values[i].append(
                        sum([
                            first_value.values[i][a] *
                            second_value.values[a][j]
                            for a in range(first_value.columns)
                        ]))
            return Matrix(values=new_values)
        raise NotImplementedError

    def transpose(self) -> Matrix:
        return Matrix(values=[[self.values[i][j] for i in range(self.lines)]
                              for j in range(self.columns)])

    def edit(self, index: tuple, value: int or float) -> None:
        self.values[index[0]][index[1]] = value

    def edit(self, indices: list[tuple], values: list[int or float]) -> None:
        for i in range(len(indices)):
            self.values[indices[i][0]][indices[i][1]] = values[i]

    def __repr__(self) -> str:
        return f"{self.lines}x{self.columns}-Matrix"

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


if __name__ == "__main__":
    m = Matrix(values=[[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    h = Matrix(values=[[1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1],
                       [0, 0, 1, 1, 0], [0, 1, 1, 0, 0]])
    print(Matrix.kernel_multiplicate(h, m))
