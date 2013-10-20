import google
import practice
import copy

def main():
	# n1 = practice.LLNode(1)
	# n2 = practice.LLNode(1)
	# n3 = practice.LLNode(1)
	# n4 = practice.LLNode(2)
	# n5 = practice.LLNode(2)
	# n6 = practice.LLNode(2)
	# n7 = practice.LLNode(4)
	# n1.next = n2
	# n2.next = n3
	# n3.next = n4
	# n4.next = n5
	# n5.next = n6
	# n6.next = n7
	# removeDuplicatesFromSortedLinkedList1(n1)
	# google.displayLL(n1)


	# mapping = {'2':['a', 'b', 'c'], '3':['d', 'e', 'f']}
	# print letterCombinationsOfAPhoneNumber("23322", mapping)

	# root = constructBinaryTreeFromInorderAndPreorderT([1,2,3,4,5,6,7], [4,2,1,3,6,5,7])
	# root = constructBinaryTreeFromInorderAndPostorderT([1,2,3,4,5,6,7], [1,3,2,5,7,6,4])
	# columns = [0] * GRIDSIZE
	# results = []
	# placeQueens(0, columns, results)
	# print results
	# print regularExpressionMatching("aab", "c*a*b*")
	# print add(4,5)
	# st = SuffixTree("bibs")
	# for i in st.root.children:
	# print st.search('ibs')
	print longestCommonSubstring("abacd", "bacef")


def removeNthNodeFromEndOfList(head, n):
	p1 = p2 = head
	for i in range(n - 1):
		p1 = p1.next
	while (p1.next != None):
		p1 = p1.next
		p2 = p2.next
	p2.next = p2.next.next
	return head

def letterCombinationsOfAPhoneNumber(digits, mapping):
	if (len(digits) == 0):
		return []
	if (len(digits) == 1):
		return mapping[digits[0]]
	first = digits[0]
	previous = letterCombinationsOfAPhoneNumber(digits[1:], mapping)
	chars = mapping[first]
	result = []
	for char in chars:
		for i in previous:
			result.append(char + i)
	return result

def combinationSum(array, target):
	if (array == [] or target < 0):
		return []
	if (len(array) == 1 and target % array[0] == 0):
		return [[array[0]] * target / array[0]]
	current = array[0]
	previous = []
	for i in range(target / current):
		previous += combinationSum()

def removeDuplicatesFromSortedLinkedList1(head):
	p1 = head
	if (p1 == None):
		return
	p2 = head.next
	while (p2 != None):
		if (p1.data == p2.data):
			p1.next = p2.next
			p2 = p2.next
		else:
			p1 = p1.next
			p2 = p2.next

def constructBinaryTreeFromInorderAndPreorderT(inorder, preorder):
	if (inorder == [] or preorder == []):
		return None
	rootVal = preorder[0]
	root = practice.Node(rootVal)
	rootIndex = inorder.index(rootVal)
	root.left = constructBinaryTreeFromInorderAndPreorderT(inorder[0:rootIndex], preorder[1 : rootIndex + 1])
	root.right = constructBinaryTreeFromInorderAndPreorderT(inorder[rootIndex + 1 : ], preorder[rootIndex + 1 : ])
	return root

def constructBinaryTreeFromInorderAndPostorderT(inorder, postorder):
	if (inorder == [] or postorder == []):
		return None
	print (inorder, postorder)
	rootVal = postorder[-1]
	root = practice.Node(rootVal)
	rootIndex = inorder.index(rootVal)
	root.left = constructBinaryTreeFromInorderAndPostorderT(inorder[0 : rootIndex], postorder[0 : rootIndex])
	root.right = constructBinaryTreeFromInorderAndPostorderT(inorder[rootIndex + 1 : ], postorder[rootIndex : -1])
	return root

def NQueens():
	result = []
	columns = [0] * GRIDSIZE
	placeQueens(0, columns, result)
	return result

def placeQueens(row, columns, result):
	if (row == GRIDSIZE):
		result.append(copy.copy(columns))
	else:
		for column in range(GRIDSIZE):
			if (checkValid(row, column, columns)):
				columns[row] = column
				placeQueens(row + 1, columns, result)

def checkValid(row, column, columns):
	for row2 in range(row):
		if (columns[row2] == column):
			return False
		if (abs(columns[row2] - column) == row - row2):
			return False
	return True

# # N queens
# GRID_SIZE = 4
# def placeQueens(row, columns, results):
# 	if (row == GRID_SIZE):
# 		results.append(copy.copy(columns))
# 	else:
# 		for col in range(GRID_SIZE):
# 			if (checkValid(columns, row, col)):
# 				columns[row] = col
# 				placeQueens(row + 1, columns, results)

# def checkValid(columns, row1, column1):
# 	for row2 in range(row1):
# 		column2 = columns[row2]
# 		if (column1 == column2):
# 			return False
# 		columnDistance = abs(column2 - column1)
# 		rowDistance = row1 - row2
# 		if (columnDistance == rowDistance):
# 			return False
# 	return True
GRIDSIZE = 4
def regularExpressionMatching(s, p):
	if (p == ""):
		if (s != ""):
			return False
		else:
			return True
	if (len(p) == 1):
		return s == p or p == "."
	if (p[1] != "*"):
		return (s[0] == p[0] or p[0] == ".") and regularExpressionMatching(s[1 : ], p[1 : ])
	else:
		# p[1] == "*"
		if (p[0] != "."):
			s_index = 0
			while ((s_index <= len(s) - 1) and (s[s_index] == p[0])):
				s_index += 1
			return regularExpressionMatching(s[s_index : ], p[2 : ])
		else:
			return len(p) == 2

def add(a, b):
	if (b == 0):
		return a
	sumVal = a ^ b
	carry = (a & b) << 1
	return add(sumVal, carry)


class SuffixTree:
	def __init__(self, s):
		self.root = SuffixTreeNode()
		for i in range(len(s)):
			suffix = s[i : ]
			self.root.insertString(suffix, i)
	def search(self, s):
		return self.root.search(s)

class SuffixTreeNode:
	def __init__(self):
		self.children = {}
		self.value = None
		self.indexes = []

	def insertString(self, s, index):
		self.indexes.append(index)
		if (s != None and len(s) > 0):
			self.value = s[0]
			child = None
			if (self.children.has_key(self.value)):
				child = self.children[self.value]
			else:
				child = SuffixTreeNode()
				self.children[self.value] = child
			remainder = s[1 : ]
			child.insertString(remainder, index)

	def search(self, s):
		if (s == None or len(s) == 0):
			return self.indexes
		else:
			first = s[0]
			if (self.children.has_key(first)):
				remainder = s[1 : ]
				return self.children[first].search(remainder)
		return None

def longestCommonSubstring(s1, s2):
	if (s1 == None or s2 == None):
		return None
	n1 = len(s1)
	n2 = len(s2)
	if (n1 == 0 or n2 == 0):
		return ""
	table = {}
	for i in range(n1):
		for j in range(n2):
			if (s1[i] == s2[j]):
				if (i > 0 and j > 0):
					table[(i, j)] = table[(i - 1, j - 1)] + 1
				else:
					table[(i, j)] = 1
			else:
				table[(i, j)] = 0
	maxLen = 0
	endIndex = 0
	for i in table.keys():
		if (table[i] > maxLen):
			maxLen = table[i]
			endIndex = i[0]
	return s1[endIndex - maxLen + 1 : endIndex + 1]


if __name__ == '__main__':
	main()