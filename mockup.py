import copy
def main():
	# print longestValidPratheses("(()")
	# print longestValidPratheses(")))))()()))()(())(()))")
	# print longestValidPratheses("()()")
	# NQueens()
	# print addBinary("11", "1")
	print search2DMatrix([[1,3,5,7], [10,11,16,20],[23,30,34,50]], 50)


def longestValidPratheses(string):
	if (string == None or string == ""):
		return 0
	stack = []
	maxLen = count = 0
	for i in string:
		if (i == "("):
			stack.append(i)
		else:
			if (stack == []):
				stack.append(i)
			elif (stack[-1] == "("):
				stack.pop()
				count += 2
			else:
				stack.append(i)
				count = 0
		maxLen = max(maxLen, count)
	return maxLen

GRID_SIZE = 8
def NQueens():
	results = []
	column = [0] * GRID_SIZE
	placeQueens(results, column, 0)
	print results

def placeQueens(results, column, row):
	if (row == GRID_SIZE):
		results.append(copy.copy(column))
		return
	for col in range(GRID_SIZE):
		if (checkValid(row, col, column)):
			column[row] = col
			placeQueens(results, column, row + 1)

def checkValid(row, col, column):
	for row2 in range(row):
		col2 = column[row2]
		if (col2 == col):
			return False
		if (abs(row2 - row) == abs(col2 - col)):
			return False
	return True

def addBinary(a, b):
	if (a == None or b == None):
		return None
	carry = 0
	result = []
	i = len(a) - 1
	j = len(b) - 1
	while (i >= 0 or j >= 0):
		addedValue = carry
		if (i >= 0):
			addedValue += int(a[i])
		if (j >= 0):
			addedValue += int(a[j])
		if (addedValue <= 1):
			carry = 0
		else:
			carry = 1
			addedValue = 0
		result.insert(0, str(addedValue))
		i -= 1
		j -= 1
	if (carry > 0):
		result.insert(0, str(carry))
	return "".join(result)

def search2DMatrix(matrix, target):
	if (matrix == None or target == None or matrix == [] or matrix == [[]] * len(matrix)):
		return False
	row = len(matrix)
	column = len(matrix[0])
	midCol = column / 2
	i = 0
	while (i < row - 1 and matrix[i][midCol] < target):
		i += 1
	if (matrix[i][midCol] == target):
		return True
	leftBottom = [matrix[k][0 : midCol] for k in range(i, len(matrix))]
	rightUp = [matrix[k][midCol + 1 : ] for k in range(0, i)]
	return search2DMatrix(leftBottom, target) or search2DMatrix(rightUp, target)


if __name__ == '__main__':
	main()