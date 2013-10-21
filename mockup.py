import copy
import practice
def main():
	# print longestValidPratheses("(()")
	# print longestValidPratheses(")))))()()))()(())(()))")
	# print longestValidPratheses("()()")
	# NQueens()
	# print addBinary("11", "1")
	# print search2DMatrix([[1,3,5,7], [10,11,16,20],[23,30,34,50]], 50)
	# print restoreIPAddress("25525511135")
	# print generatePrantheses(4)
	# plusOne([1,2,3,4,5])
	# print climbStairs(4)
	# print longestCommonPrefix(["aaaaabb", "aaaaaabbb"])
	print searchForRange([1,2,3,4,5,5,5,5,5,6,7,8], 5)


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
	rightUp = [matrix[k][midCol + 1 : ] for k in range(0, i + 1)]
	return search2DMatrix(leftBottom, target) or search2DMatrix(rightUp, target)

def restoreIPAddress(s):
	result = []
	if (s == None or len(s) == 0):
		return result
	depth = start = 0
	ip = ""
	generate(s, start, depth, result, ip)
	return result

def generate(s, start, depth, result, ip):
	if ((len(s) - start) > (4 - depth) * 3):
		return
	if (len(s) - start < (4 - depth)):
		return
	if (depth == 4):
		ip = ip[0 : len(ip) - 1]
		if (ip not in result):
			result.append(ip)
		return
	num = 0
	for i in range(start, min(start + 3, len(s))):
		num = num * 10 + int(s[i])
		if (num <= 255):
			generate(s, i + 1, depth + 1, result, ip + str(num) + ".")
		if (num == 0):
			break

def generatePrantheses(n):
	if (n == 0):
		return []
	if (n == 1):
		return ["()"]
	previous = generatePrantheses(n - 1)
	result = []
	for i in previous:
		result.append("()" + i)
		for k in range(len(i)):
			if (i[k] == '('):
				result.append(i[0 : k + 1] + "()" + i[k + 1 : ])
	return result

def convertSortedArrayToBST(array):
	if (array == None or array == []):
		return None
	mid = len(array) / 2
	root = practice.Node(array[mid])
	root.left = convertSortedArrayToBST(array[0 : mid])
	root.right = convertSortedArrayToBST(array[mid + 1 : ])
	return root

def plusOne(digits):
	if (digits == None or digits == []):
		return
	carry = 0
	for i in range(len(digits) - 1, -1, -1):
		sumVal = digits[i] + carry
		if (sumVal >= 10):
			carry = 1
			sumVal = sumVal % 10
		else:
			carry = 0
		digits[i] = sumVal
	if (carry > 0):
		digits.insert(0, carry)

def climbStairs(n):
	p = q = 1
	for i in range(2, n + 1):
		temp = q
		q += p
		p = temp
	return q

first = None
second = None
pre = None
def recoverTree(root):
	first = None
	second = None
	pre = None
	inorder(root)
	first.data, second.data = second.data, first.data
	return

def inorder(root):
	if (root == None):
		return
	else:
		inorder(root.left)
		if (pre == None):
			pre = root
		else:
			if (pre.data > root.data):
				if (first == None):
					first = pre
				second = root
			pre = root
		inorder(root.right)


def longestCommonPrefix(strs):
	if (strs == None or strs == []):
		return ""
	rightMost = len(strs[0]) - 1
	for i in range(1, len(strs)):
		for j in range(rightMost + 1):
			if (strs[i][j] != strs[0][j]):
				rightMost = j - 1
	return strs[0][0 : rightMost + 1]

def searchForRange(array, target):
	lower = 0
	upper = len(array)
	while (lower < upper):
		mid = (lower + upper) / 2
		if (array[mid] < target):
			lower = mid + 1
		else:
			upper = mid
	if (array[lower] != target):
		return (-1, -1)
	upper = len(array)
	while (lower < upper):
		mid = (lower + upper) / 2
		if (array[mid] > target):
			upper = mid
		else:
			lower = mid + 1
	return (lower, upper - 1)

class KeyValue:
	def __init__(self, key, value):
		self.key = key
		self.value = value
class HashTable:
	TABLESIZE = 20
	def __init__(self):
		self.dataStore = [[]] * TABLESIZE

	def get(self, key):
		# get api
		hashValue = self.hash(key)
		for record in self.dataStore[hashValue]:
			if (record.key == key):
				return record.value


	def set(self, key, value):
		# set api
		hashValue = self.hash(key)
		keyValue = KeyValue(key, value)
		self.dataStore[hashValue].append(keyValue)



	def hash(self, key):
		total = 0
		for i in key:
			total += ord(i)
		return total % TABLESIZE



if __name__ == '__main__':
	main()