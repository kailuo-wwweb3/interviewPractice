import heapq
import practice
# import treesAndGraphs

def main():
	# print largestSubArrayWithEqual0sAnd1s([0,1,0,1,0,1])
	# print largestSubArrayWithEqual0sAnd1s([1,1,1,1,1,1])
	# print twoSum([1,2,3,4,5], 6)
	# print decodeURL("le.%20   d")
	# print findKmin([1,2,3,4,5], 2)
	# print partition_string("aababcabcd")
	# print longestEvenPalindrome("abcicbbcdefggfed")
	# print practice.longestPalindromicSubstring("abcicbbcdefggfed")
	# root = practice.convertSortedArrayToBST([1,2,3,4,5])
	# result = serializeBST(root)
	# print result
	# root = deserializeBST(result)
	# treesAndGraphs.inorderTraversal()
	# print mergeKSortedArrays([[1,3,5,7],[2,4,6,8], [0,9,10,11]])
	# print findPeakNumber([1, 3, 20, 4, 1, 0])
	# printMatrixDiagonally([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]])
	# addGreaterValuesToEveryNodeInBST(root)
	# treesAndGraphs.inorderTraversal(root)
	# print root.right.data
	# root = pruneUtil(root, 7)
	# print "----"
	# treesAndGraphs.inorderTraversal(root)
	# print findMinInSortedAndRotatedArray([8,9,10,1,2,3,4,5,6,7])
	print findElementInSortedAndRotatedArray([5,6,7,8,9,1,2,3,4], 7)


def largestSubArrayWithEqual0sAnd1s(array):
	count_ones = count_zeros = 0
	for item in array:
		if item == 1:
			count_ones += 1
		else:
			count_zeros += 1
	if (abs(count_ones - count_zeros) == len(array)):
		return 0, []
	i = 0
	j = len(array) - 1
	while i < j:
		diff = count_ones - count_zeros
		if diff == 0:
			break
		elif diff > 0:
			if array[i] == 1:
				i += 1
				count_ones -= 1
			elif array[j] == 1:
				j -= 1
				count_ones -= 1
			else:
				i += 1
				count_zeros -= 1
		elif array[i] == 0:
			i += 1
			count_zeros -= 1
		elif array[j] == 0:
			j -= 1
			count_zeros -= 1
		else:
			j -= 1
			count_ones -= 1
	return j - i + 1, array[i : j + 1]

def twoSum(array, n):
	hashTable = {}
	result = []
	for element in array:
		if (not hashTable.has_key(n - element)):
			hashTable[element] = True
		else:
			result.append((element, n - element))
	return result

def decodeURL(url):
	result = ""
	i = 0
	while i < len(url):
		if (url[i:i + 3] == "%20"):
			result += " "
			i += 3
		else:
			result += url[i]
			i += 1
	return result

def findKmin(array, k):
	heap = []
	for element in array:
		heapq.heappush(heap, element)
		if (len(heap) == k + 1):
			heap.remove(heap[len(heap) - 1])
	print heap
	return heap[len(heap) - 1]

def partition_string(my_str):
	phrase_list = []
	index = 0

	while index < len(my_str):
		cur_phrase = my_str[index]

		while (index + 1 < len(my_str)) and (cur_phrase in phrase_list):
			cur_phrase += my_str[index + 1]
			index += 1
		phrase_list.append(cur_phrase)
		index += 1

	return phrase_list

def longestEvenPalindrome(string):
	dict_record = {}
	max_Length = 0
	startingIndex = 0
	print len(string)
	print "--------"
	for i in range(len(string)):
		for j in range(len(string)):
			dict_record[(i, j)] = False
	for i in range(len(string) - 1):
		if (string[i] == string[i + 1]):
			dict_record[(i, i + 1)] = True
			max_Length = 2
			startingIndex = i
	for length in range(4, len(string)):
		for i in range(len(string) - length + 1):
			j = i + length - 1
			if (i == 8):
				print string[i:]
				print length
				print string[j]
			if ((string[i] == string[j]) and (dict_record[(i + 1, j - 1)])):
				dict_record[(i, j)] = True
				max_Length = length
				startingIndex = i
	return string[startingIndex : startingIndex + max_Length]

# practice more!!!
def inorderTraversalWithoutRecur(root):
	current = root
	stack = []
	while (current != None):
		if (current.left != None):
			stack.append(current)
			current = current.left
		else:
			print current.data
			current = current.right
			while (current == None and len(stack) != 0):
				current = stack.pop()
				print current.data
				current = current.right

def preorderTraversalWithoutRecur(node):
	stack = []
	while (len(stack) != 0) or (node != None):
		if (node != None):
			print node.data
			if node.right != None:
				stack.append(node.right)
			node = node.left
		else:
			node = stack.pop()

def countInsideRange(root, a, b):
	if (root == None):
		return 0
	else:
		if (root.data <= a):
			return countInsideRange(root.right, a, b)
		elif (root.data >= b):
			return countInsideRange(root.left, a, b)
		else:
			return 1 + countInsideRange(root.left, a, b) + countInsideRange(root.right, a, b)

# def constructBSTFromInorderAndPreorder(inorder, preorder):
	# if (inorder == [] or preorder == []):
		# return None
	# rootData = preorder[0]
	# root = practice.Node(rootData)
	# rootIndex = inorder.index(rootData)

def printInLevelOrder(root):
	queue = []
	level = 0
	queue.append((root, 0))
	bufferList = []
	while (queue != []):
		current = queue.pop(0)
		if (current[1] != level):
			print bufferList
			bufferList = [current[0].data]
			level += 1
		else:
			bufferList.append(current[0].data)
		if (current[0].left != None):
			queue.append((current[0].left, current[1] + 1))
		if (current[0].right != None):
			queue.append((current[0].right, current[1] + 1))
	print bufferList

def serializeBST(root):
	result = []
	serializeBST_helper(root, result)
	return result

def serializeBST_helper(node, result):
	result.append(node.data) if node != None else result.append("#")
	if node != None:
		serializeBST_helper(node.left, result)
		serializeBST_helper(node.right, result) 

def deserializeBST(result):
	if (result[0] == "#"):
		result.pop(0)
		return None
	else:
		root = practice.Node(result.pop(0))
		root.left = deserializeBST(result)
		root.right = deserializeBST(result)
	return root


def mergeKSortedArrays(arrays):
	heap = []
	result = []
	for i in range(len(arrays)):
		heapq.heappush(heap, (arrays[i].pop(0), i))
	while (heap != []):
		minElement = heapq.heappop(heap)
		result.append(minElement[0])
		index = minElement[1]
		if (arrays[index] != []):
			heapq.heappush(heap, (arrays[index].pop(0), index))
	return result
def findMinInSortedAndRotatedArray(array):
	return findMinInSortedAndRotatedArray_helper(array, 0, len(array) - 1)

def findMinInSortedAndRotatedArray_helper(array, low, high):
	if (low == high):
		return array[low]
	mid = (low + high) / 2
	if (mid < high and array[mid + 1] < array[mid]):
		return array[mid + 1]

	if (mid > low and array[mid] < array[mid - 1]):
		return array[mid]
	if (array[high] > array[mid]):
		return findMinInSortedAndRotatedArray_helper(array, low, mid - 1)
	return findMinInSortedAndRotatedArray_helper(array, mid + 1, high)

def findPeakNumber(array):
	return findPeakNumber_helper(array, 0, len(array) - 1)

def findPeakNumber_helper(array, start, end):
	mid = (start + end) / 2
	if ((mid == 0 or array[mid - 1] <= array[mid]) and (mid == len(array) - 1 or array[mid + 1] <= array[mid])):
		return array[mid]
	elif (mid > 0 and array[mid - 1] > array[mid]):
		return findPeakNumber_helper(array, start, mid - 1)
	else:
		return findPeakNumber_helper(array, mid + 1, end)

def printMatrixDiagonally(matrix):
	row = len(matrix)
	column = len(matrix[0])
	diagon = []
	for i in range(row + column - 1):
		(currentX, currentY) = (i,0) if i <= row - 1 else (row - 1, i % (row - 1))
		while ((currentX >= 0) and (currentY <= column - 1)):
			diagon.append(matrix[currentX][currentY])
			currentX -= 1
			currentY += 1
		print diagon
		diagon = []

def addGreaterValuesToEveryNodeInBST(root):
	inorder = []
	constructInorder(root, inorder)
	for i in range(len(inorder) - 1, 0, -1):
		inorder[i - 1].data += inorder[i].data


def constructInorder(node, inorder):
	if (node.left != None):
		constructInorder(node.left, inorder)
	inorder.append(node)
	if (node.right != None):
		constructInorder(node.right, inorder)

def pruneUtil(root, k):
	return pruneUtil_helper(root, k, 0)[0]

def pruneUtil_helper(node, k, sumVal):
	if (node == None):
		return (None, sumVal)
	leftSum = sumVal + node.data
	rightSum = leftSum
	leftSide = pruneUtil_helper(node.left, k, leftSum)
	rightSide = pruneUtil_helper(node.right, k, rightSum)
	node.left = leftSide[0]
	node.right = rightSide[0]
	maxVal = max(leftSide[1], rightSide[1])
	if (maxVal < k):
		return (None, maxVal)
	else:
		return (node, maxVal)

def findElementInSortedAndRotatedArray(array, key):
	return findElementInSortedAndRotatedArray_helper(array, 0, len(array) - 1, key)


def findElementInSortedAndRotatedArray_helper(array, low, high, key):
	if (low == high):
		if (array[low] != key):
			return False
	mid = (low + high) / 2
	if (array[mid] == key):
		return True
	if ((array[low] < key and array[mid] > key) or (array[mid] < key and array[low] < key and array[mid] < array[low])):
		return findElementInSortedAndRotatedArray_helper(array, low, mid - 1, key)
	return findElementInSortedAndRotatedArray_helper(array, mid + 1, high, key)

if __name__ == '__main__':
	main() 