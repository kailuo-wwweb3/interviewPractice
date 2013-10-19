import heapq
import copy
import practice
import random
from collections import defaultdict
def main():
	# print findIntersectionOfTwoLists([1,2,3,4,5],[1,3,5,9,10])
	# print validParentheses("()[()()({}{}[{}{}]{})()]{}}}")
	# print validPalidrome("race a car")
	# print fourSum([1,2,3,4,5,6], 13)
	# print pow_imp(3,3)
	# print wordLadder1("hit", "cog", ["hot","dot","dog","lot","log"])
	# print mergeIntervals([(1,3), (2,4), (5,7), (6,8)])
	# print removeElement([1,2,3,4,3,3], 3)
	# print insertInterval([[1,3],[6,9]], [2,5])
	# l1 = practice.LLNode(1)
	# l2 = practice.LLNode(2)
	# l3 = practice.LLNode(3)
	# l4 = practice.LLNode(4)
	# l1.next = l2
	# l2.next = l3
	# l3.next = l4
	# head = swapNodesInPair(l1)
	# displayLL(head)
	# print combinations(10, 2)
	# root = practice.convertSortedArrayToBST([1,2,3,4,5])
	# print sumRootToLeafNumbers(root)
	"""
	1  4  7  11 15
	2  5  8  12 19
	3  6  9  16 22
	10 13 14 17 24
	18 21 23 26 30
	"""
	# print search2DMatrix([[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], 30)
	# print cropMatrix([[1,2,3],[1,2,3],[1,2,3]], 1,2,1,2)
	# print findMedianOfAnArray([2,1,4,3, 8, 7,5,6])
	# print addBinary("11", "1")
	# testTrie()
	# print longestCommonSubstring("ABAB", "BABA")
	# print reverseArrayOfWordsInPlace([1,2,3,4])
	# columns = [0] * GRID_SIZE
	# results = []
	# placeQueens(0, columns, results)
	# print results
	# print uniquePaths(3,7)
	# print uniquePath_dp(3, 7)
	root = practice.convertSortedArrayToBST([1,2,3,4,5,6,7,8,9,10])
	# print inorderSuccessor(root.left.left).data
	# print decodeWays("12101123432")
	# print maximumSubArray([-2,1,-3,4,-1,2,1,-5,4])
	# minimumPathSum([[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], 5, 5)
	# print bestTimeToBuyAndShareStock([2,3,4,1,3,7])
	# print root.right.data
	# displayFlattenTree(flattenTree(root))
	# print minimumTriangle([10, 30, 20, 101, 102, 103, 10000, 10001, 10002])
	# print romanToInteger("MMX|||")
	# print exp(4,3)
	# print palindromPartition("aab")
	# print regularExpressionMatching("aa", "a")
	# print regularExpressionMatching("aa", "aa")
	# print regularExpressionMatching("aaa", "aa")
	# print regularExpressionMatching("aa", "a*")
	# print regularExpressionMatching("aa", ".")
	# print regularExpressionMatching("ab", ".*")
	# print regularExpressionMatching("aab", "c*a*b")
	# populateNextRightPointer2(root)
	# print divideTwoIntegers(100, 5)
	# ls = [6,7,8,9,10,11,1,2,3,4,5]
	# print searchInRSA(ls, 2)
	# for i in ls:
		# print searchInRSA(ls, i)
	# print searchForRange([1,4,4,4,5,6,7], 4)
	# print threeSum([-25, -10, -7, -3, 2, 4, 8, 10], -25)
	# print generateParentheses(4)
	# print strStr("", "asdf")
	# array = [1,2,3,3,4,5]
	# print removeDuplicatesFromSortedArray(array)
	# print sqrt(3)
	# print pathSum(root, 12)
	# print reverseInteger(123)
	printBSTLevelByLevel(root)



class Trie(object):
	def __init__(self):
		self.root = defaultdict(Trie)
		self.value = None

	def add(self, s, value):
		head, tail = s[0], s[1:]
		cur_node = self.root[head]
		if not tail:
			cur_node.value = value
			return
		cur_node.add(tail, value)

	def lookup(self, s, default=None):
		head, tail = s[0], s[1:]
		node = self.root[head]
		if tail:
			return node.lookup(tail)
		return node.value or default

	def remove(self, s):
		head, tail = s[0], s[1:]
		if head not in self.root:
			return False
		node = self.root[head]
		if tail:
			return node.remove(tail)
		else:
			del node
			return True

	def prefix(self, s):
		if not s:
			return True
		head, tail = s[0], s[1:]
		if head not in self.root:
			return False
		node = self.root[head]
		return node.prefix(tail)

	def items(self):
		for char, node in self.root.iteritems():
			if node.value == None:
				yield node.items
			else:
				yield node

#union-find algorithm implementation
class UFElement(object):
	def __init__(self, value):
		self.value = value
		self.parent = None
		self.rank = 0


def makeSet(x):
	x.parent = x
	x.rank = 0

def union(x, y):
	xRoot = find(x)
	yRoot = find(y)
	if (xRoot == yRoot):
		return
	if xRoot.rank > yRoot.rank:
		yRoot.parent = xRoot
	elif xRoot.rank < yRoot.rank:
		xRoot.parent = yRoot
	else:
		xRoot.parent = yRoot
		yRoot.rank += 1
def find(x):
	while (x.parent != x):
		x = x.parent
	return x


def testTrie():
	strings = ["A", "to", "tea", "ted", "ten", "i", "in", "inn"]
	trie = Trie()
	for i in strings:
		trie.add(i, i)
	print trie.root.values()


def longestCommonSubstring(a, b):
	length_A = len(a)
	length_B = len(b)
	table = {}
	for i in range(length_A):
		for j in range(length_B):
			if (a[i] == b[j]):
				if (i > 0 and j > 0):
					table[(i, j)] = table[(i - 1, j - 1)] + 1
				else:
					table[(i, j)] = 1
			else:
				table[(i, j)] = 0
	max_Length = 0
	ending_index_A = 0
	for i in range(length_A):
		for j in range(length_B):
			if (table[(i, j)] > max_Length):
				max_Length = table[(i, j)]
				ending_index_A = i
	return a[i - max_Length + 1 : i + 1]


def displayLL(head):
	while head != None:
		print head.data
		head = head.next


def findIntersectionOfTwoLists(l1, l2):
	p1 = p2 = 0
	result = []
	while ((p1 < len(l1)) and (p2 < len(l2))):
		if (l1[p1] > l2[p2]):
			p2 += 1
		elif (l1[p1] < l2[p2]):
			p1 += 1
		else:
			result.append(l1[p1])
			p1 += 1
			p2 += 1
	return result

def validParentheses(string):
	if (len(string) <= 1):
		return False
	stack = [string[0]]
	for i in range(1, len(string)):
		if (string[i] == ']'):
			if (stack != [] and stack[-1] == '['):
				stack.pop()
			else:
				return False
		elif (string[i] == ')'):
			if (stack != [] and stack[-1] == '('):
				stack.pop()
			else:
				return False
		elif (string[i] == '}'):
			if (stack != [] and stack[-1] == '{'):
				stack.pop()
			else:
				return False
		else:
			stack.append(string[i])
	return True

def climbStairs(n):
	return climbStairs_helper(n, {0:1, 1:1})

def climbStairs_helper(n, table):
	if (n in table):
		return table[n]

	table[n] = climbStairs_helper(n - 2) + climbStairs_helper(n - 1)
	return table[n]

def validPalidrome(string):
	i = 0
	j = len(string) - 1
	while (i <= j):
		while (notValid(string[i])):
			i += 1
		while (notValid(string[j])):
			j -= 1
		if (string[i].lower() == string[j].lower()):
			i += 1
			j -= 1
		else:
			return False
	return True

def notValid(char):
	alphabets = []
	a = ord('a')
	for i in range(26):
		alphabets.append(chr(a + i))
	A = ord('A')
	for i in range(26):
		alphabets.append(chr(A + i))
	return not char in alphabets

class FourSumRecord(object):
	def __init__(self, value1, value2, start, end):
		self.value1 = value1
		self.value2 = value2
		self.start = start
		self.end = end
	
	def _cmp_(self, obj):
		return (self.value1 + self.value2) - (obj.value1 + obj.value2)

	def getSum(self):
		return self.value1 + self.value2
	def notCommonWith(self, obj):
		return self.start != obj.start and self.end != obj.end and self.start != obj.end and self.end != obj.start


def fourSum(array, k):
	ls = []
	result = []
	for i in range(len(array) - 1):
		for j in range(i + 1, len(array)):
			ls.append(FourSumRecord(array[i], array[j], i, j))
	ls.sort()
	i = 0
	j = len(ls) - 1
	while (i < j):
		if (ls[i].getSum() + ls[j].getSum() == k and ls[i].notCommonWith(ls[j])):
			result.append((ls[i].value1, ls[i].value2, ls[j].value1, ls[j].value2))
			i += 1
			j -= 1
		elif (ls[i].getSum() + ls[j].getSum() < k):
			i += 1
		else:
			j -= 1
	return result

def pow_imp(x, n):
	if n == 0:
		return 1.0
	if n % 2 == 0:
		return pow_imp(x, n / 2) * pow_imp(x, n / 2)
	if n % 2 == 1:
		return pow_imp(x, n / 2) * pow_imp(x, n / 2) * x


class WordLadderRecord(object):
	def __init__(self, word):
		self.word = word
		self.prev = None

	def _eq_(self, obj):
		return self.word == obj.word
		
def wordLadder1(start, end, dictionary):
	startNode = WordLadderRecord(start)
	queue = [startNode]
	added = [start]
	while (queue != []):
		current = queue.pop(0)
		currentWord = current.word
		for i in range(len(currentWord)):
			currentWord_copy = currentWord
			currentWord_copy_list = list(currentWord_copy)
			fuckChar = currentWord_copy_list[i]
			for j in range(26):
				currentWord_copy_list[i] = chr(ord('a') + j)
				stringVersion = ''.join(currentWord_copy_list)
				if (stringVersion == end):
					adj = WordLadderRecord(stringVersion)
					adj.prev = current
					return wordLadder1_displayPath(adj)
				elif ((stringVersion in dictionary) and (stringVersion not in added)):
					adj = WordLadderRecord(stringVersion)
					adj.prev = current
					queue.append(adj)
					added.append(stringVersion)
	return []

def wordLadder1_displayPath(node):
	result = []
	while (node != None):
		result.insert(0,node.word)
		node = node.prev
	return result




def wordLadder_helper(string1, string2):
	count = 0
	for i in range(len(string1)):
		if (string1[i] != string2[i]):
			count += 1
	if count == 1:
		return True
	else:
		return False

class Interval(object):
	def __init__(self, start, end):
		self.start = start
		self.end = end
	def _cmp_(self, obj):
		return self.start - obj.start

def mergeIntervals(intervals):
	for i in range(len(intervals)):
		intervals[i] = Interval(intervals[i][0], intervals[i][1])
	intervals.sort()
	stack = [intervals[0]]
	for i in range(1, len(intervals)):
		current = intervals[i]
		top = stack[-1]
		if (current.start > top.end):
			stack.append(current)
		elif (current.end > top.end):
			top.end = current.end
	return [(interval.start, interval.end) for interval in stack]

def removeElement(array, element):
	length = len(array)
	i = j = 0
	while (i < len(array)):
		if (array[i] != element):
			array[j] = array[i]
			j += 1
		i += 1
	return j

def insertInterval(intervals, insert):
	for i in range(len(intervals) - 1):
		if (intervals[i][0] < insert[0] and intervals[i + 1][0] > insert[0]):
			print i
			break
	stack = intervals[0:i + 1]
	if (insert[0] > stack[-1][1]):
		stack.append(insert)
	elif (insert[1] > stack[-1][1]):
		stack[-1][1] = insert[1]
	for j in range(i + 1, len(intervals)):
		print intervals[j]
		if (intervals[j][0] > stack[-1][1]):
			stack.append(intervals[j])
		elif (intervals[j][0] > stack[-1][1]):
			stack[-1][1] = intervals[j][1]
	return stack

def swapNodesInPair(head):
	if (head.next.next == None):
		head.next.next = head
		head = head.next
		head.next.next = None
		return head
	swaped = swapNodesInPair(head.next.next)
	head.next.next = head
	head = head.next
	head.next.next = swaped
	return head

def combinations(n, k):
	result = []
	for i in range(1, n - k + 2):
		result += combinations_helper(n, k, i)
	return result

def combinations_helper(n, k, current):
	if (k == 1):
		return [[current]]
	result = []
	for i in range(current + 1, n + 1):
		ls = combinations_helper(n, k - 1, i)
		for j in ls:
			j.insert(0, current)
		result += ls
	return result

def sumRootToLeafNumbers(root):
	stack = [root]
	visited = {root:"visited"}
	sumVal = 0
	while (stack != []):
		top = stack[-1]
		if (top.left == None and top.right == None):
			sumVal += convertListToNumber(stack)
			stack.pop()
		elif (top.left != None and (not visited.has_key(top.left))):
			stack.append(top.left)
			visited[top.left] = "visited"
		elif (top.right != None and (not visited.has_key(top.right))):
			stack.append(top.right)
			visited[top.right] = "visited"
		else:
			stack.pop()
	return sumVal

def convertListToNumber(ls):
	num = ""
	for i in ls:
		num += str(i.data)
	return int(num)

def search2DMatrix(matrix, target):
	if (matrix == [] or matrix == [[]] or matrix == [[], []] or matrix == [[], [], []]):
		return False
	row = len(matrix)
	column = len(matrix[0])
	if (target < matrix[0][0] or target > matrix[row - 1][column - 1]):
		return False
	mid = column / 2
	i = 0
	while (i <= row - 1 and matrix[i][mid] <= target):
		if (matrix[i][mid] == target):
			return True
		i += 1
	return search2DMatrix(cropMatrix(matrix, i, row - 1, 0, mid - 1), target) or search2DMatrix(cropMatrix(matrix, 0, i - 1, mid + 1, column - 1), target)

def cropMatrix(matrix, r1, r2, c1, c2):
	result = []
	matrix = matrix[r1:r2 + 1]
	for i in matrix:
		result.append(i[c1:c2 + 1])
	return result

def findMedianOfAnArray(array):
	length = len(array)
	if (length % 2 == 0):
		return (quickSelect(array, length / 2) + quickSelect(array, length / 2 - 1)) / 2.0
	else:
		return quickSelect(array, length / 2)

def quickSelect(array, k):
	random.shuffle(array)
	low = 0
	high = len(array) - 1
	i = partition(array, low, high)
	while (True):
		if (i == k):
			return array[k]
		elif (i < k):
			low = i + 1
		else:
			high = i - 1
		i = partition(array, low, high)

def partition(array, low, high):
	i, j = low + 1, high
	while (True):
		while (array[i] < array[low]):
			if (i == high):
				break
			i += 1
		while (array[j] > array[low]):
			if (j == low):
				break
			j -= 1
		if (i >= j):
			break
		array[i], array[j] = array[j], array[i]
	array[j], array[low] = array[low], array[j]
	return j

def addBinary(a, b):
	result = []
	i, j = len(a) - 1, len(b) - 1
	print i, j
	carry = 0

	while ((i >= 0) or (j >= 0)):
		sumVal = carry
		if (i >= 0):
			sumVal += int(a[i])
			i -= 1
		if (j >= 0):
			sumVal += int(b[j])
			j -= 1
		if (sumVal == 0 or sumVal == 1):
			carry = 0
		else:
			carry = 1
			sumVal = 0
		result.insert(0, str(sumVal))
		print result
	if (carry == 1):
		result.insert(0, str(carry))
	return "".join(result)

def reverseArrayOfWordsInPlace(array):
	i, j = 0, len(array) - 1
	while i <= j:
		array[i], array[j] = array[j], array[i]
		i += 1
		j -= 1
	return array

def decodeWays(s):
	n = len(s)
	if (n == 0):
		return 0
	c = [0] * n
	c[0] = 1
	for i in range(1, n + 1):
		c1 = 0
		if (s[i - 1] != '0'):
			c1 = c[i - 1]
		c2 = 0
		if (i >= 2 and (s[i - 2] == '1' or s[i - 2] == '2' and s[i - 1] <= '6')):
			c2 = c[i - 2]
		c[i] = c1 + c2
	return c[n]

# N queens
GRID_SIZE = 4
def placeQueens(row, columns, results):
	if (row == GRID_SIZE):
		results.append(copy.copy(columns))
	else:
		for col in range(GRID_SIZE):
			if (checkValid(columns, row, col)):
				columns[row] = col
				placeQueens(row + 1, columns, results)

def checkValid(columns, row1, column1):
	for row2 in range(row1):
		column2 = columns[row2]
		if (column1 == column2):
			return False
		columnDistance = abs(column2 - column1)
		rowDistance = row1 - row2
		if (columnDistance == rowDistance):
			return False
	return True

def uniquePaths(m, n):
	table = {}
	paths = uniquePaths_helper(0, 0, m, n, table)
	print table
	return paths

def uniquePaths_helper(currentX, currentY, m, n, table):
	if (table.has_key((currentX, currentY))):
		return table[(currentX, currentY)]
	if (currentX == m - 1 and currentY == n - 1):
		table[(currentX, currentY)] = 1
		return 1
	if (currentX > m - 1 or currentY > n - 1):
		return 0
	table[(currentX, currentY)] = uniquePaths_helper(currentX + 1, currentY, m, n, table) + uniquePaths_helper(currentX, currentY + 1, m, n, table)
	return table[(currentX, currentY)]

def uniquePath_dp(m, n):
	table = {}
	for i in range(m):
		table[(i, n - 1)] = 1
	for i in range(n):
		table[(m - 1, i)] = 1
	for i in range(m - 2, -1, -1):
		for j in range(n - 2, -1, -1):
			table[(i, j)] = table[(i + 1, j)] + table[(i, j + 1)]
	return table[(0, 0)]

def inorderSuccessor(n):
	if (n.right != None):
		current = n.right
		while (current.left != None):
			current = current.left
		return current
	else:
		while (n.parent != None and n.parent.left != n):
			n = n.parent
		return n



def decodeWays(s):
	n = len(s)
	if n == 0:
		return 0
	record = [0] * (n + 1)
	record[0] = 1
	for i in range(1, n + 1):
		# consider for one digit
		c1 = c2 = 0
		if (s[i - 1] != '0'):
			c1 = record[i - 1]
		if ((i >= 2) and ((s[i - 2] == '1') or (s[i - 2] == '2' and s[i - 1] <= '6'))):
			c2 = record[i - 2]
		record[i] = c1 + c2
	print record
	return record[n]

def maximumSubArray(array):
	maxSoFar = maxSum = 0
	for i in array:
		maxSoFar += i
		maxSoFar = max(maxSoFar, 0)
		maxSum = max(maxSoFar, maxSum)
	return maxSum

def minimumPathSum(matrix, m, n):
	table = {}
	for i in range(m - 2, -1, -1):
		table[(i, n - 1)] = matrix[i][n - 1] + matrix[i + 1][n - 1]
	for i in range(n - 2, -1, -1):
		table[(m - 1, i)] = matrix[m - 1][i] + matrix[m - 1][i + 1]
	for i in range(m - 2, -1, -1):
		for j in range(n - 2, -1, -1):
			table[(i, j)] = min(table[(i + 1, j)], table[(i, j + 1)]) + matrix[i][j]
	print table


def bestTimeToBuyAndShareStock(array):
	smallest = pSmallest = (array[0], 0)
	largest = (array[1], 1)
	for i in range(2, len(array)):
		if (array[i] > largest[0]):
			largest = (array[i], i)
			if (pSmallest[1] > smallest[1]):
				smallest = pSmallest
		if (array[i] < smallest[0]):
			pSmallest = (array[i], i)
	return (smallest[0], largest[0])

def flattenTree(root):
	if (root == None):
		return None
	left = root.left
	right = copy.copy(root.right)
	if (left != None):
		root.right = flattenTree(left)
		root.left = None
	if (right != None):
		current = root.right
		while (current.right != None):
			current = current.right
		current.right = flattenTree(right)
	return root

def displayFlattenTree(root):
	while (root != None):
		print root.data
		root = root.right

class ValueIndex(object):
	def __init__(self, value, index):
		self.value = value
		self.index = index
	def _cmp_(self, obj):
		return self.value - obj.value

def minimumTriangle(array):
	n = len(array)
	if (n < 3):
		return []
	array.sort()
	for i in range(len(array)):
		array[i] = ValueIndex(array[i], i)
	a = array[0]
	b = array[1]
	c = array[2]
	while (True):
		if (a.value + b.value > c.value):
			return [a.value, b.value, c.value]
		else:
			if ((a.index == b.index - 1) or (c.index < n - 1 and (array[c.index + 1].value - b.value < array[a.index + 1].value - a.value))):
				b = ValueIndex(array[c.index + 1].value, c.index + 1)
				b, c = c, b
			elif ((c.index == n - 1) or (array[c.index + 1].value - b.value > array[a.index + 1].value - a.value)):
				a = ValueIndex(array[a.index + 1].value, a.index + 1)
			else:
				break
	return []

def romanToInteger(string):
	table = {'|':1, 'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000}
	result = 0
	for i in string:
		result += table[i]
	return result

def exp(x, n):
	if (n == 0):
		return 1
	if (n == 1):
		return x
	if (n % 2 != 0):
		return exp(x, n / 2) * exp(x, n / 2) * x
	else:
		return exp(x, n / 2) * exp(x, n / 2)


def palindromPartition(string):
	if (string == None or len(string) == 0):
		return []
	isPal = {}
	for i in range(len(string)):
		isPal[(i, i)] = True
	for i in range(len(string) - 1):
		if (string[i] == string[i + 1]):
			isPal[(i, i + 1)] = True
	return palindromPartitionHelper(string, 0, isPal)

def palindromPartitionHelper(string, start, isPal):
	pa = []
	if (start > len(string) - 1):
		return [[]]
	for i in range(start, len(string)):
		if (isPal.has_key((start, i)) and isPal[(start, i)]):
			for subPal in palindromPartitionHelper(string, i + 1, isPal):
				subPal.append(string[start : i + 1])
				pa.append(subPal)
	return pa

def regularExpressionMatching(s, p):
	if (p == ""):
		return s == ""
	if ((len(p) >= 2 and p[1] != '*') or (len(p) == 1)):
		return (p[0] == s[0] or p[0] == '.') and regularExpressionMatching(s[1:], p[1:])
	else:
		index_s = 0
		while (index_s < len(s) and ((p[0] == s[index_s]) or (p[0] == '.'))):
			if (regularExpressionMatching(s[index_s:], p[2:])):
				return True
			index_s += 1
		return regularExpressionMatching(s[index_s:], p[2:])

def combinationSum(ls, target):
	result = []
	combinationSum_helper(ls, target, result)
	return result

# not constant space.
def populateNextRightPointer(root):
	setUpLevels(root, 0)
	queue = [root]
	previous = None
	while (queue != []):
		current = queue.pop(0)
		if (previous != None and previous.level == current.level):
			previous.next = current
		if (current.left != None):
			queue.append(current.left)
		if (current.right != None):
			queue.append(current.right)
		previous = current
	return root

def setUpLevels(node, level):
	if (node == None):
		return
	node.level = level
	setUpLevels(node.left, level + 1)
	setUpLevels(node.right, level + 1)

# with constant space
def populateNextRightPointer2(root):
	leftWall = root
	while (leftWall != None):
		across = leftWall
		while (across != None):
			if (across.left != None):
				across.left.next = across.right
			if (across.right != None and across.next != None):
				across.right.next = across.next.left
			across = across.next
		leftWall = leftWall.left

# a / b
def divideTwoIntegers(a, b):
	if (a < b):
		return 0
	return 1 + divideTwoIntegers(a - b, b)

def searchInRSA(array, k):
	return searchInRSA_helper(array, k, 0, len(array) - 1)

def searchInRSA_helper(array, k, start, end):
	if (start >= end):
		if (array[end] == k):
			return True
		else:
			return False
	mid = (start + end) / 2
	if (array[mid] == k):
		return True
	if ((array[mid] > k and k >= array[start]) or (array[mid] < array[end] and (k < array[mid] or k > array[end]))):
		return searchInRSA_helper(array, k, start, mid)
	else:
		return searchInRSA_helper(array, k, mid + 1, end)

def searchForRange(ls, target):
	return searchForRange_helper(ls, target, 0, len(ls) - 1)

def searchForRange_helper(array, target, start, end):
	if (array[start] == array[end] == target):
		return (start, end)
	mid = (start + end) / 2
	if (array[mid] < target):
		return searchForRange_helper(array, target, mid + 1, end)
	elif (array[mid] > target):
		return searchForRange_helper(array, target, start, mid - 1)
	else:
		head = tail = mid
		while ((array[head - 1] == target and head > start) or (array[tail + 1] == target and tail < end)):
			if (array[head - 1] == target):
				head -= 1
			if (array[tail + 1] == target):
				tail += 1
		return (head, tail)

def stringToInteger(string):
	if (string == None):
		return None
	p = 0
	while (string[p] == ' '):
		p += 1
	sign = 1
	if (string[p] == '-'):
		sign = -1
		p += 1
	if (string[p] == '+'):
		sign = 1
		p += 1
	num = 0
	while (ord(string[p]) >= 48 and ord(string[p]) <= 57):
		if ((num == 214748364 and string[p] > '7') or (num > 214748364)):
			pass
		num = 10 * num + int(string[p])
		p += 1
	return sign * num

def threeSum(array, val):
	result = []
	array.sort()
	n = len(array)
	for i in range(n - 3):
		a = array[i]
		k = i + 1
		l = n - 1
		while (k < l):
			b = array[k]
			c = array[l]
			if (a + b + c == val):
				result.append((a,b,c))
				k += 1
				l -= 1
			elif (a + b + c > val):
				l -= 1
			else:
				k += 1
	return result

def generateParentheses(n):
	storeSet = []
	if (n == 0):
		storeSet.append("")
	else:
		previousSet = generateParentheses(n - 1)
		for element in previousSet:
			# insert front
			if (("()" + element) not in storeSet):
				storeSet.append("()" + element)
			for i in range(len(element)):
				if (element[i] == '('):
					contResult = element[0:i + 1] + "()" + element[i + 1 : ]
					if (contResult not in storeSet):
						storeSet.append(contResult)
	return storeSet

def strStr(text, target):
	text_length = len(text)
	target_length = len(target)
	for i in range(text_length - target_length):
		match = 0
		for p in range(i, i + target_length):
			if (text[p] == target[p - i]):
				match += 1
		if (match == target_length):
			return i
	return -1

def removeDuplicatesFromSortedArray(array):
	n = len(array)
	i = 0
	if (n <= 1):
		return n
	for j in range(1, n):
		if (array[j] != array[i]):
			i += 1
			array[i] = array[j]
	return i + 1

def sqrt(x):
	i = 0
	j = x / 2 + 1
	while (i <= j):
		mid = (i + j) / 2
		sq = mid * mid
		if (sq == x):
			return mid
		elif (sq < x):
			i += mid
		else:
			j = mid - 1
	return j

def pathSum(root, sumVal):
	stack = [(root, sumVal - root.data)]
	visited = [root]
	while (stack != []):
		top = stack[-1]
		node = top[0]
		remaining = top[1]
		if (node.left != None and node.left not in visited):
			stack.append((node.left, remaining - node.left.data))
		elif (node.right != None and node.right not in visited):
			stack.append((node.right, remaining - node.right.data))
		else:
			if (remaining == 0):
				return [i[0].data for i in stack]
			else:
				stack.pop()
		visited.append(node)
	return []

def reverseInteger(integer):
	sign = 1 if integer >= 0 else -1
	integer = abs(integer)
	if (integer == 0):
		return 0
	j = k = 1
	while (integer / k > 0):
		k *= 10
	k /= 10
	result = 0
	while (k >= 1):
		result = (integer / k) * j + result
		integer = integer % k
		k /= 10
		j *= 10
	return sign * result

def printBSTLevelByLevel(root):
	if (root == None):
		return
	currentLevel = [root]
	nextLevel = []
	while (currentLevel != []):
		current = currentLevel.pop(0)
		print current.data,
		if (current.left != None):
			nextLevel.append(current.left)
		if (current.right != None):
			nextLevel.append(current.right)
		if (currentLevel == []):
			currentLevel, nextLevel = nextLevel, currentLevel
			print '\n',



if __name__ == '__main__':
	main()