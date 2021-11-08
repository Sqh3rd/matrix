class Matrix:
	def __init__(self, lines:int = 2, columns:int = 2, values:list = [[0,0],[0,0]]):
		if len(values) < 1:
			values = [[0]]
		elif len(values[0]) < 1:
			for val in values:
				val = [0]
		if lines < 1:
			lines = 1
		if columns < 1:
			columns = 1
		if values != [[0,0],[0,0]]:
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

	def add(self, x):
		if not type(x) is Matrix:
			raise NotImplementedError
		if not len(x.values) == len(self.values) and len(x.values[0]) == len(self.values[0]):
			raise NotImplementedError
		return Matrix(values=[[x.values[i][c] + d for c, d in enumerate(self.values[i])] for i in range(len(self.values))])

	def sub(self, first_val, second_val):
		if type(first_val) is Matrix and type(second_val) is Matrix:
			first_val = first_val.values
			second_val = second_val.values
			if not len(first_val) == len(second_val) and len(first_val[0]) == len(second_val[0]):
				raise NotImplementedError
			return Matrix(values=[[first_val[i][c] - d for c, d in enumerate(second_val[i])] for i in range(len(second_val))])
		raise NotImplementedError

	def mul(self, first_value, second_value):
		if (type(first_value) is int or type(first_value) is float) or (type(second_value) is int or type(second_value) is float):
			if type(first_value) is Matrix:
				return Matrix(values=[[val * second_value for val in innerval] for innerval in first_value.values])
			elif type(second_value) is Matrix:
				return Matrix(values=[[val * first_value for val in innerval] for innerval in second_value.values])
		elif type(first_value) is Matrix and type(second_value) is Matrix:
			if not first_value.columns == second_value.lines:
				raise Exception("The first Matrix must have the same amount of columns as the amount of lines the second Matrix has!")
			new_values = []
			for i in range(first_value.lines):
				new_values.append([])
				for j in range(second_value.columns):
					new_values[i].append(sum([first_value.values[i][a] * second_value.values[a][i] for a in range(first_value.columns)]))
			return Matrix(values = new_values)
		raise NotImplementedError

	def transpose(self):
		return Matrix(values = [[self.values[i][j] for i in range(self.lines)] for j in range(self.columns)])
		
	def __repr__(self):
		return f"{self.lines}x{self.columns}-Matrix"

	def __str__(self):
		strings = [['|'.join([str(b) for b in a])] for a in self.values]
		return '\n'.join([str(string).replace('[', '').replace(']', '').replace("'", '') for string in strings])

	def __add__(self, x):
		return self.add(x)

	def __radd__(self, x):
		return self.add(x)

	def __sub__(self, x):
		return self.sub(self, x)

	def __rsub__(self, x):
		return self.sub(x, self)

	def __mul__(self, x):
		return self.mul(self, x)

	def __rmul__(self, x):
		return self.mul(x, self)

	def __eq__(self, x):
		if not type(x) is Matrix:
			return False
		if not len(x.values) == len(self.values) and len(x.values[0]) == len(self.values[0]):
			return False
		for i, line in enumerate(self.values):
			for j, val in enumerate(line):
				if not val == x.values[i][j]:
					return False
		return True

if __name__ == '__main__':
	m = Matrix(values = [[2,2],[3,3]])
	b = Matrix(values = [[4,4],[5,5]])
	c = m.transpose()
	print(c)