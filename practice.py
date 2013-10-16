import copy
import operator
import math


def main():
	# print isPalindrome(101)
	# print longestPalindromicSubstring('abc')
	# print longestSubStringWithNonRepeatingChar('ab')
	# print permutations('ab')
	# print generateSubsets([1, 2, 3])
	# history = []
	# print generateAllPermutationsOfNpairOfParen(4)
	# print makeChanges(25, 25)
	# print magicIndex([-40, -20, -1, 1, 2, 3, 5, 7, 9, 12, 13], 0, 10)
	# print countWays(4)
	# print TwoSum([1, 2], 3)
	# print ThreeSum([-1, 0, 1, 2, -1, -4])
	# array = [0,1,2,3,4,5,6,7,8,9]
	# root = convertSortedArrayToBST(array)
	# print (root.left.left.data, root.left.data, root.left.right.data, root.data, root.right.data, root.right.left.data, root.right.right.data)
	# printInOrder(root)
	# preorder = [3, 5, 6, 2, 7, 4, 1, 0, 8]
	# inorder = [6, 5, 7, 2, 4, 3, 0, 1, 8]
	# root = constructBSTfromInorderAndPreorder(inorder, preorder)
	# printInOrder(root)
	# print '------------'
	# printPreOrder(root)
	# print 'LCA is:'
	# print LCA1(root, 5, 1).data
	# newNode = insertIntoCyclicSortedList(createAnCyclicSortedList(), 3.1)
	# printCyclicList(newNode)
	# print isBST(root)
	# print MaximumValueContiguousSubSequence([-1, 2, 3, 1, -3, -2, 100])
	# logName = 'log.txt'
	# top10Log(logName)
	# print isPerfectSquare(90)
	# print stringCompress("aabbccddddddddd")
	# rotateImageBy90([[1,1,1,1], [2,2,2,2], [3,3,3,3], [4,4,4,4]])
	# ifAnElementIs0ThenSetEntireRowAndColumnToBe0([[0,1,1,0],[2,2,0,1],[0,0,0,1]])

	print KMP_search("addddddbcdefghijk", "bcd")

class Node:
	def __init__(self, value):
		self.data = value
		self.left = None
		self.right = None
		self.parent = None
		self.next = None
		self.level = 0

		
class LLNode:
	def __init__(self, value):
		self.data = value
		self.next = None
		self.prev = None


def isPalindrome(number):
	shortDiv = 10
	longDiv = 10
	while number / longDiv >= 10:
		longDiv = longDiv * 10
	while not number == 0:
		front = number / longDiv
		end = number % shortDiv
		if not front == end:
			return False
		number = (number % longDiv) / 10
		longDiv = longDiv / 100
	return True

def longestSubStringWithNonRepeatingChar(s):
	n = len(s)
	i = 0
	j = 0
	maxLen = 0
	exist = {}
	while j < n:
		if exist.has_key(s[j]) and exist[s[j]]:
			maxLen = max(maxLen, j - i)
			while not s[i] == s[j]:
				exist[s[i]] = False
				i = i + 1
			i = i + 1
			j = j + 1
		else:
			exist[s[j]] = True
			j = j + 1
	maxLen = max(maxLen, n - i)
	return maxLen

def longestPalindromicSubstring(s):
	table = {}
	maxLen = 0
	starting = 0
	for i in range(len(s)):
		for j in range(len(s)):
			table[(i, j)] = False
	for i in range(len(s)):
		table[(i, i)] = True
		maxLen = 1
		starting = i
	for i in range(len(s) - 1):
		if s[i] == s[i + 1]:
			table[(i, i + 1)] = True
			maxLen = 2
			starting = i
	for length in range(3, len(s)):
		for i in range(len(s) - length):
			j = i + length - 1
			if s[i] == s[j] and table[(i + 1, j - 1)]:
				maxLen = length
				table[(i, j)] = True
				starting = i
	return s[starting:starting + maxLen]

def permutations(s):
	if len(s) == 0:
		return []
	elif len(s) == 1:
		return [s]
	previous = permutations(s[1:len(s)])
	current = s[0]
	result = []
	for i in previous:
		for k in range(len(i) + 1):
			result.append(i[0:k] + current + i[k:])
	return result

def generateSubsets(ls):
	if len(ls) == 0:
		return [ls]
	elif len(ls) == 1:
		return [[], ls]
	current = ls[0]
	previous = generateSubsets(ls[1:])
	previous_copy = copy.copy(previous)
	for i in previous:
		i_copy = copy.copy(i)
		i_copy.append(current)
		previous_copy.append(i_copy)
	return previous_copy

history = []

def generateAllPermutationsOfNpairOfParen(n):
	if n == 0:
		return []
	elif n == 1:
		return ['()']
	previous = generateAllPermutationsOfNpairOfParen(n - 1)
	result = []
	for i in previous:
		i_copy = copy.copy(i)
		i_copy = '()' + i_copy
		if not i_copy in history:
			history.append(i_copy)
			result.append(i_copy)
		for j in range(len(i)):
			i_copy = copy.copy(i)
			if i[j] == '(':
				i_copy = i_copy[0:j + 1] + '()' + i_copy[j + 1:]
				if not i_copy in history:
					history.append(i_copy)
					result.append(i_copy)
	return result

def makeChanges(n, denom):
	next_denom = 0
	if denom == 25:
		next_denom = 10
	elif denom == 10:
		next_denom = 5
	elif denom == 5:
		next_denom = 1
	elif denom == 1:
		return 1
	ways = 0
	i = 0
	while i * denom <= n:
		ways = ways + makeChanges(n - i * denom, next_denom)
		i = i + 1
	return ways

def magicIndex(ls, start, end):
	length = len(ls)
	mid = (start + end) / 2
	print ls, mid
	if ls[mid] < mid:
		return magicIndex(ls, mid + 1, end)
	elif ls[mid] > mid:
		return magicIndex(ls, start, end - 1)
	else:
		return mid

def countWays(n):
	if n < 0:
		return 0
	elif n == 0:
		return 1
	else:
		return countWays(n - 1) + countWays(n - 2) + countWays(n - 3)


def TwoSum(array, k):
	result = []
	array.sort()
	first = 0
	last = len(array) - 1
	while first <= last:
		if k - array[last] > array[first]:
			first = first + 1
		elif k - array[first] < array[last]:
			last = last - 1
		if array[last] + array[first] == k:
			result.append((array[first], array[last]))
			first = first + 1
			last = last - 1
	return result


def ThreeSum(array):
	array.sort()
	result = []
	n = len(array)
	for i in range(n):
		j = i + 1
		k = n - 1
		while j < k:
			sum_two = array[i] + array[j]
			if sum_two + array[k] < 0:
				j = j + 1
			elif sum_two + array[k] > 0:
				k = k - 1
			else:
				data = (array[i], array[j], array[k])
				if not data in result:
					result.append(data)
				j = j + 1
				k = k - 1
	return result

def convertSortedArrayToBST(array):
	n = len(array)
	if n == 1:
		return Node(array[0])
	elif n == 0:
		return None
	else: 
		mid = n / 2
		root = Node(array[mid])
		root.left = convertSortedArrayToBST(array[0:mid])
		root.right = convertSortedArrayToBST(array[mid + 1:])
	return root

def printInOrder(root):
	if not root.left == None:
		printInOrder(root.left)
	print root.data
	if not root.right == None:
		printInOrder(root.right)

def printPreOrder(root):
	print root.data
	if not root.left == None:
		printPreOrder(root.left)
	if not root.right == None:
		printPreOrder(root.right)

def constructBSTfromInorderAndPreorder(inorder, preorder):
	if len(inorder) == 0:
		return None
	if len(inorder) == 1:
		return Node(inorder[0])
	rootValue = preorder[0]
	rootIndex = inorder.index(rootValue)
	root = Node(rootValue)
	root.left = constructBSTfromInorderAndPreorder(inorder[0:rootIndex], preorder[1:rootIndex + 1])
	root.right = constructBSTfromInorderAndPreorder(inorder[rootIndex + 1:], preorder[rootIndex + 1:])
	return root

def LCA(root, p, q):
	if root == None:
		return None
	if root.data == p or root.data == q:
		return root
	L = LCA(root.left, p, q)
	R = LCA(root.right, p, q)
	if (not L == None) and (not R == None):
		return root
	elif L == None:
		return R
	else:
		return L

# O(h)
def LCA1(root, p, q):
	if root == None:
		return None
	if max(p, q) < root.data:
		return LCA1(root.left, p, q)
	elif min(p, q) > root.data:
		return LCA1(root.right, p, q)
	else:
		return root


def createAnCyclicSortedList():
	node1 = LLNode(1)
	node2 = LLNode(2)
	node3 = LLNode(3)
	node4 = LLNode(4)
	node5 = LLNode(5)
	node1.next = node2
	node2.next = node3
	node3.next = node4
	node4.next = node5
	node5.next = node1
	return node1

def insertIntoCyclicSortedList(node, x):
	newNode = LLNode(x)
	while 1:
		nextNode = node.next
		if newNode.data < nextNode.data and node.data < newNode.data:
			break
		if node.data > nextNode.data and newNode.data > node.data:
			break
		node = node.next
	# insert after node
	newNode.next = node.next
	node.next = newNode
	return newNode

def printCyclicList(node):
	result = []
	while 1:
		if not node.data in result:
			result.append(node.data)
		else:
			break
		node = node.next
	print result

def isBST(root):
	return isBSTHelper(root, -float('inf'), float('inf'))

def isBSTHelper(root, minValue, maxValue):
	if root == None:
		return True
	if root.data < minValue or root.data > maxValue:
		return False
	if (not isBSTHelper(root.left, minValue, root.data)) or (not isBSTHelper(root.right, root.data, maxValue)):
		return False
	return True

def MaximumValueContiguousSubSequence(array):
	maxSoFar = 0
	maxEndingHere = 0
	for s in array:
		maxEndingHere = maxEndingHere + s
		maxEndingHere = max(maxEndingHere, 0)
		maxSoFar = max(maxSoFar, maxEndingHere)
	return maxSoFar

# def findAllAnagrams(fileName):
# 	inputFile = open(fileName, 'r')
# 	lines = inputFile.readlines()
# 	ls = []
# 	history = {}
# 	for line in lines:
# 		for s in line:
# 			ls.append[s]
# 		ls_sort = ls
# 		ls_sort.sort()
# 		if not ls_sort in history.keys():
# 			history[ls_sort] = [ls]
# 		else:
# 			history[ls_sort].append(ls)
# 	return history


def top10Log(logName):
	f = open(logName, 'r')
	records = {}
	lines = f.readlines()
	for line in lines:
		content = line[:-1]
		if not records.has_key(content):
			records[content] = 1
		else:
			records[content] = records[content] + 1
	sorted_records = sorted(records.iteritems(), key=operator.itemgetter(1))
	# sorted_records = sorted(records.iterkeys())
	print sorted_records[len(sorted_records) - 5:]
	return

def isPerfectSquare(n):
	a = 0
	odd = 1
	while a < n:
		a = a + odd
		if a == n:
			return True
		odd = odd + 2
	return False


def stringCompress(s):
	result = ""
	currentChar = s[0]
	count = 1
	for i in range(1, len(s)):
		if s[i] == currentChar:
			count = count + 1
		else:
			result = result + (currentChar + str(count))
			count = 1
			currentChar = s[i]
	result = result + (currentChar + str(count))
	if len(result) >= len(s):
		return s
	else:
		return result

def rotateImageBy90(matrix):
	n = len(matrix)
	for layer in range(n/2):
		first = layer
		last = n - 1 - layer
		for i in range(first, last):
			offset = i - first

			top = matrix[first][i]
			matrix[first][i] = matrix[last - offset][first]
			matrix[last - offset][first] = matrix[last][last - offset]
			matrix[last][last - offset] = matrix[i][last]
			matrix[i][last] = top
	print matrix
	return

def ifAnElementIs0ThenSetEntireRowAndColumnToBe0(matrix):
	row = []
	column = []
	for i in range(len(matrix)):
		row.append(False)
	for i in range(len(matrix[0])):
		column.append(False)

	for i in range(len(matrix)):
		for j in range(len(matrix[0])):
			if matrix[i][j] == 0:
				row[i] = True
				column[j] = True
	for i in range(len(matrix)):
		for j in range(len(matrix[0])):
			if (row[i] and column[j]):
				matrix[i][j] = 0
	print matrix


def KMP_search(text, search_string):
	next = preComputeTable(search_string)
	j = 0
	for i in range(len(text)):
		if (text[i] == search_string[j]):
			j += 1
		else:
			j = next[j]
		if (j == len(search_string)):
			return i - len(search_string) + 1


def preComputeTable(search_string):
	next = {}
	for i in range(len(search_string)):
		next[i] = 0
	x = 0
	for j in range(len(search_string)):
		if (search_string[x] == search_string[j]):
			next[j] = next[x]
			x += 1
		else:
			next[j] = x + 1
			x = next[x]
	return next

if __name__ == '__main__':
	main()