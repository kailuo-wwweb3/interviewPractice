import practice
import copy

identity_matrix = [[1,0],[0,1]]
def main():
	# print isPerfectSquare(1)
	# print findSubPermutation("abcdefg", "ba")
	# root = practice.convertSortedArrayToBST([1,2,3,4,5,6,7,8,9,10])
	# print returnLevelWithMaxNodes(root)
	# print getNumOfWaysToCalculateATargetNumber([2,4,6,8], 0, 12)
	# print longestSubStringWithEqual0sAnd1s("11111111101111111111")
	# print findMaxSumSubSequence([-6,2,-1,2,11])
	# print_perimeter(root)
	# print root.data
	# practice.printInOrder(root)
	# reverseBT(root)
	# practice.printInOrder(root)
	# print findMinInRSA([6,7,8,1,2,3,4,5])
	# print longestPalindrome("babcbabcbaccba")
	# print findLargestSubset([1,6,10,4,7,9,5, 5,8])
	# root = practice.Node(1)
	# root.left = practice.Node(1)
	# root.left.left = practice.Node(1)
	# root.right = practice.Node(1)
	# root.right.right = practice.Node(1)
	# root.right.right.left = practice.Node(1)
	# print findDeepest1(root)
	# print isAnagram("baa", "a")
	# print continousLargestSum([6,-1, 3, 0, 4])
	# string = ['a','b','a','c','d']
	# removeDuplicateCharInString(string)
	# print string
	# print findSumCombinations([5, 5, 10, 2, 3], 15)
	# print func(10)
	# print mergeSort([3,4,2,1,5])
	# test = ['a','1','b','2','c','4']
	# convertToOriginalString(test)
	# print test
	# head1 = constructLL([1,2,3])
	# head2 = constructLL([8,9,1])
	# print head1.next.data
	# print head1
	# sumHead = sumOfLinkedList(head1, head2)
	# printLL(sumHead)
	ls = [1,-2,3,-4,5]
	mutateList(ls)
	print ls
	print "---"


def printLL(head):
	while head != None:
		print head.data
		head = head.next

def constructLL(array):
	i = 1
	head = practice.LLNode(array[0])
	current = head
	while (i < len(array)):
		current.next = practice.LLNode(array[i])
		i += 1
		current = current.next
	return head

def isPerfectSquare(n):
	current = 0
	odd = 1
	while (True):
		if (current > n):
			return False
		elif (current == n):
			return True
		else:
			current = current + odd
			odd = odd + 2

def findSubPermutation(string, tmp_string):
	string = "".join(sorted(string))
	tmp_string = "".join(sorted(tmp_string))
	if (len(string) < len(tmp_string)):
		return False
	for i in range(len(string) - len(tmp_string)):
		if (string[i:i + len(tmp_string)] == tmp_string):
			return True
	return False

def returnLevelWithMaxNodes(root):
	hashTable = {}
	returnLevelWithMaxNodes_helper(root, hashTable, 0)
	max_num = hashTable[0]
	for i in hashTable.keys():
		if (hashTable[i] > max_num):
			max_num = hashTable[i]
	return max_num


def returnLevelWithMaxNodes_helper(node, hashTable, level):
	if (node == None):
		return
	if (not hashTable.has_key(level)):
		hashTable[level] = 1
	else:
		hashTable[level] += 1
	returnLevelWithMaxNodes_helper(node.left, hashTable, level + 1)
	returnLevelWithMaxNodes_helper(node.right, hashTable, level + 1)


def findLastElementOfTheLoopInLinkedList(head):
	slowPointer = head
	fastPointer = head
	# find a loop in linked list
	while (fastPointer != None or fastPointer.next != None):
		slowPointer = slowPointer.next
		fastPointer = fastPointer.next.next
		if (slowPointer == fastPointer):
			break
	if (fastPointer == None or fastPointer.next == None):
		return None
	slowPointer = head
	while (slowPointer != fastPointer):
		slowPointer = slowPointer.next
		fastPointer = fastPointer.next
	return slowPointer

def getNumOfWaysToCalculateATargetNumber(array, index, target):
	if (index == len(array) and target != 0):
		return 0
	elif (index == len(array)):
		return 1
	return getNumOfWaysToCalculateATargetNumber(array, index + 1, target) + getNumOfWaysToCalculateATargetNumber(array, index + 1, target - array[index]) + getNumOfWaysToCalculateATargetNumber(array, index + 1, target + array[index])

def longestSubStringWithEqual0sAnd1s(string):
	number_of_1s = 0
	number_of_0s = 0
	for element in string:
		if (element == '0'):
			number_of_0s += 1
		elif (element == '1'):
			number_of_1s += 1
	i = 0
	j = len(string) - 1
	while i < j:
		if (number_of_0s == number_of_1s):
			return string[i : j + 1]
		elif (number_of_0s > number_of_1s):
			if (string[i] == '0'):
				i += 1
			elif (string[j] == '0'):
				j -= 1
			else:
				i += 1
			number_of_0s -= 1
		else:
			if (string[i] == '1'):
				i += 1
			elif (string[j] == '1'):
				j -= 1
			else:
				j -= 1
			number_of_0s -= 1
	return ""

def findMaxSumSubSequence(array):
	maxSoFar = 0
	maxEndingHere = 0
	for element in array:
		maxEndingHere += element
		maxEndingHere = max(0, maxEndingHere)
		maxSoFar = max(maxSoFar, maxEndingHere)
	return maxSoFar

def print_perimeter(root):
	print_perimeter_helper(root, 0, 0)

def print_perimeter_helper(node, left, right):
	if (node == None):
		return
	if (right == 0):
		print node.data
		print_perimeter_helper(node.left, left + 1, right)
	if (node.left == None and node.left == None and right > 0):
		print node.data
	print_perimeter_helper(node.right, left, right + 1)
	if (left == 0):
		if (right != 0):
			print node.data

def reverseBT(root):
	tmp = root.left
	root.left = root.right
	root.right = tmp
	if (root.left != None):
		reverseBT(root.left)
	if (root.right != None):
		reverseBT(root.right)
def findMinInRSA(array):
	return findMinInRSA_helper(array, 0, len(array) - 1)

def findMinInRSA_helper(array, start, end):
	if (start == end):
		return array[start]
	if (array[start] < array[end]):
		return array[start]
	middle = (start + end) / 2
	small1 = min(array[start], array[middle])
	small2 = min(array[middle + 1], array[end])
	return findMinInRSA_helper(array, start, middle) if (small1 < small2) else findMinInRSA_helper(array, middle + 1, end)

# longest palindromic substring o(n)
def preProcess(s):
	n = len(s)
	if (n == 0):
		return "^$"
	ret = "^"
	for i in range(n):
		ret += "#" + s[i:i+1]
	ret += "#$"
	return ret

def longestPalindrome(s):
	T = preProcess(s)
	n = len(T)
	center = 0
	R = 0
	table = {}
	for i in range(n):
		table[i] = 0
	for i in range(1, n - 1):
		i_mirror = 2 * center - i
		table[i] = min(R - i, table[i_mirror]) if R > i else 0
		while (T[i + 1 + table[i]] == T[i - 1 - table[i]]):
			table[i] += 1

		if (i + table[i] > R):
			center = i
			R = i + table[i]

	maxLen = 0
	centerIndex = 0
	for i in range(1, n - 1):
		if (table[i] > maxLen):
			maxLen = table[i]
			centerIndex = i
	return s[(centerIndex - 1 - maxLen) / 2 : (centerIndex - 1 + maxLen) / 2]

def findLargestSubset(arr):
	table = {}
	first = 0
	last = 0
	for i in arr:
		beg = end = i
		if i in table:
			continue
		table[i] = "existed"
		if i - 1 in table:
			beg = table[i - 1]
		if i + 1 in table:
			end = table[i + 1]
		table[beg] = end
		table[end] = beg
		if end - beg > last - first:
			first = beg
			last = end
	return list(range(first, end + 1))

def findDeepest1(root):
	if root.data == 0:
		return []
	path = []
	return findDeepest1_helper(root, path)

def findDeepest1_helper(node, path):
	path.append(node.data)
	path_copy_left = copy.copy(path)
	path_copy_right = copy.copy(path)
	if (node.left != None and node.left.data == 1):
		path_copy_left = findDeepest1_helper(node.left, path_copy_left)
	if (node.right != None and node.right.data == 1):
		path_copy_right = findDeepest1_helper(node.right, path_copy_right)
	return path_copy_left if len(path_copy_left) > len(path_copy_right) else path_copy_right

def isAnagram(string1, string2):
	if (len(string1) != len(string2)):
		return False
	ls = []
	for i in range(256):
		ls.append(0)
	for i in range(len(string1)):
		ls[ord(string1[i])] += 1
	for i in range(len(string2)):
		ls[ord(string2[i])] -= 1
		if ls[ord(string2[i])] < 0:
			return False
	return True

def continousLargestSum(array):
	maxSoFar = 0
	maxSumEnding = 0
	start_tmp = 0
	end_tmp = 0
	start = 0
	end = 0
	for i in range(len(array)):
		element = array[i]
		maxSoFar += element
		if (maxSoFar < 0):
			maxSoFar = 0
			if (start_tmp < len(array) - 1):
				start_tmp = i + 1
		else:
			end_tmp = i
		if (maxSumEnding < maxSoFar):
			maxSumEnding = maxSoFar
			start = start_tmp
			end = end_tmp
	return (maxSumEnding, start, end)

def removeDuplicateCharInString(string):
	if (string == None):
		return
	if (len(string) < 2):
		return
	tail = 1
	for i in range(1, len(string)):
		for j in range(tail):
			if (string[i] == string[j]):
				break
		j += 1
		if (tail == j):
			string[tail] = string[i]
			tail += 1
def findSumCombinations(array, sumVal):
	if (array == []):
		return 0
	if ((len(array) == 1 and array[0] == sumVal) or (sumVal == 0)):
		return 1
	return findSumCombinations(array[1:], sumVal) + findSumCombinations(array[1:], sumVal - array[0])

def func(n):
	hashTable = {1:1, 2:3}
	return func_helper(n, hashTable)

def func_helper(n, hashTable):
	if (hashTable.has_key(n)):
		return hashTable[n]
	hashTable[n] = 2 * (func_helper(n - 1, hashTable) + func_helper(n - 2, hashTable))
	return hashTable[n]

def mergeSort(array):
	if (len(array) <= 1):
		return array
	mid = len(array) / 2
	left = mergeSort(array[0:mid])
	right = mergeSort(array[mid:])
	return merge(left, right)

def merge(left, right):
	result = []
	i = 0
	j = 0
	left_length = len(left)
	right_length = len(right)
	while ((i < left_length) and (j < right_length)):
		if (left[i] < right[j]):
			result.append(left[i])
			i += 1
		else:
			result.append(right[j])
			j += 1
	if (i < left_length):
		for k in range(i, left_length):
			result.append(left[k])
	if (j < right_length):
		for k in range(j, right_length):
			result.append(right[k])
	return result

def convertToOriginalString(array):
	for i in range(1, len(array)):
		if (array[i].isdigit()):
			count = int(array[i])
			array.remove(array[i])
			for j in range(count - 1):
				array.insert(i, array[i - 1])

def sumOfLinkedList(head1, head2):
	sumSet = sumOfLinkedList_helper(head1, head2)
	if (sumSet[1] != 0):
		result_head = practice.LLNode(sumSet[1])
		result_head.next = sumSet[0]
	else:
		result_head = sumSet[0]
	return result_head



def sumOfLinkedList_helper(node1, node2):
	sumVal = 0
	if (node1 == None and node2 == None):
		return (None, -1)
	elif (node1 != None):
		sumVal += node1.data
	elif (node2 != None):
		sumVal += node2.data
	previous_set = sumOfLinkedList_helper(node1.next, node2.next)
	sumVal += previous_set[1]
	next_carry = 1 if sumVal >= 10 else 0
	sumNode = practice.LLNode(sumVal % 10)
	sumNode.next = previous_set[0]
	return (sumNode, next_carry)


def matrix_multiply(a, b):
	return [[a[0][0]*b[0][0] + a[0][1]*b[1][0], a[0][0]*b[0][1] + a[0][1]*b[1][1]], [a[1][0]*b[0][0] + a[1][1]*b[1][0], a[1][0]*b[0][1]+a[1][1]*b[1][1]]]

def matrix_power(m, n):
	result = identity_matrix
	power = m
	while n > 0:
		if n % 2 == 1:
			result = matrix_multiply(result, power)
		power = matrix_multiply(power, power)
		n = n // 2
	return result

def fib(n):
	return matrix_power([[0,1],[1,1]], n)[0][1]


def mutateList(ls):
	flag = False
	while (not flag):
		flag = True
		for i in range(len(ls) - 1):
			if (ls[i] < 0 and ls[i + 1] > 0):
				flag = False
				ls[i], ls[i + 1] = ls[i + 1], ls[i]




if __name__ == '__main__':
	main()