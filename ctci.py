import practice
import google
import copy

def main():
	# print hasAllUniqueCharacters("abcda")
	# print isPermutationEachOther("abcde", "adeab")
	# test1 = list("Mr John Smith    ")
	# relaceSpacesInString(test1)
	# print test1
	# print compressString("aabcccccaaa")

	# head1 = convertArrayToLinkedList([7,1,6])
	# head2 = convertArrayToLinkedList([5,9,2])
	# google.displayLL(head)
	# google.displayLL(removeDuplicatesFromUnsortedLinkedListWithoutBuffer(head))
	# google.displayLL(partitionLinkedList(head, 3))
	# google.displayLL(addTwoNumbers(head1, head2))
	# root = practice.convertSortedArrayToBST([1,2,3,4,5])
	# for i in createLinkedListsOfAllNodesAtEachDepth(root):
		# print [j.data for j in i]

	# node1 = practice.Node(1)
	# node2 = practice.Node(2)
	# node3 = practice.Node(3)
	# node4 = practice.Node(4)
	# node5 = practice.Node(5)
	# node1.left = node2
	# node2.parent = node1
	# node2.left = node3
	# node2.right = node4
	# node3.parent = node2
	# node4.parent = node2
	# node4.right = node5
	# node5.parent = node4

	# print inorderSuccessor(node3).data
	'''
	a a a a 
	b b b b
	c c c c
	d d d d
	'''
	matrix = [['a', 'a','a','a'], ['b','b','b','b'], ['c','c','c','c'],['d','d','d','d']]
	rotateMatrix(matrix)
	print matrix




def convertArrayToLinkedList(array):
	head = practice.LLNode(array[0])
	current = head
	i = 1
	while (i < len(array)):
		current.next = practice.LLNode(array[i])
		i += 1
		current = current.next
	return head


def hasAllUniqueCharacters(string):
	array = [False] * 256
	for i in string:
		if (array[ord(i)]):
			return False
		else:
			array[ord(i)] = True
	return True

def isPermutationEachOther(s1, s2):
	table = {}
	for i in s1:
		if (table.has_key(i)):
			table[i] += 1
		else:
			table[i] = 1
	for i in s2:
		if (not table.has_key(i)):
			return False
		else:
			table[i] -= 1
			if (table[i] < 0):
				return False
	return True

def relaceSpacesInString(charArray):
	length = len(charArray)
	i = length - 1
	while (charArray[i] == ' '):
		i -= 1
	j = length - 1
	while (j >= 0):
		if (charArray[i] == ' '):
			charArray[j - 2] = '%'
			charArray[j - 1] = '2'
			charArray[j] = '0'
			j -= 3
		else:
			charArray[j] = charArray[i]
			j -= 1
		i -= 1

def compressString(charArray):
	newCharArray = []
	count = 1
	for i in range(1, len(charArray)):
		if (charArray[i] == charArray[i - 1]):
			count += 1
		else:
			newCharArray.append(charArray[i - 1])
			newCharArray.append(str(count))
			count = 1
	newCharArray.append(charArray[i - 1])
	newCharArray.append(str(count))
	if (len(newCharArray) >= len(charArray)):
		return charArray
	return newCharArray

def removeDuplicatesFromUnsortedLinkedList(head):
	current = head
	table = {head.data : True}
	while (current != None and current.next != None):
		if (table.has_key(current.next.data)):
			current.next = current.next.next
		else:
			table[current.next.data] = True
			current = current.next
	return head

def removeDuplicatesFromUnsortedLinkedListWithoutBuffer(head):
	current = head
	while (current != None):
		currentCopy = current
		while (currentCopy.next != None):
			if (currentCopy.next.data == current.data):
				currentCopy.next = currentCopy.next.next
			else:
				currentCopy = currentCopy.next
		current = current.next
	return head

def partitionLinkedList(head, x):
	beforeStart = beforeEnd = afterStart = afterEnd = None
	current = head
	while (current != None):
		if (current.data < x):
			if (beforeStart == None):
				beforeStart = practice.LLNode(current.data)
				beforeEnd = beforeStart
			else:
				beforeEnd.next = practice.LLNode(current.data)
				beforeEnd = beforeEnd.next
		else:
			if (afterStart == None):
				afterStart = practice.LLNode(current.data)
				afterEnd = afterStart
			else:
				afterEnd.next = practice.LLNode(current.data)
				afterEnd = afterEnd.next
		current = current.next
	beforeEnd.next = afterStart
	return beforeStart

def addTwoNumbers(head1, head2):
	return addTwoNumbers_helper(head1, head2, 0)

def addTwoNumbers_helper(head1, head2, carry):
	if (head1 == None and head2 == None):
		if (carry != 0):
			return practice.LLNode(carry)
		else:
			return None
	sumVal = carry
	if (head1 != None):
		sumVal += head1.data
	if (head2 != None):
		sumVal += head2.data
	sumNode = practice.LLNode(sumVal % 10)
	nextCarry = 1 if sumVal > 10 else 0
	sumNode.next = addTwoNumbers_helper(head1.next, head2.next, nextCarry)
	return sumNode

# implement three stacks using a single array.
# array = [1,2,3,4,5,6,7,8,9]
# pointers = [-1,-1,-1]
# def push(i, element):
# 	if (pointers[i] == -1):
# 		pointers[i] = i / 3
# 	if (pointers[i] == (i + 1) / 3):

# 	else:
# 		pointers[i] += 1
# 	array[pointers[i]] = element

# def pop(i):
# 	if (pointers[i] == -1):
# 		return None
# 	top = array[pointers]
# 	pointers[i] -= 1
# 	return top

def createLinkedListsOfAllNodesAtEachDepth(root):
	lists = []
	createLinkedListsOfAllNodesAtEachDepth_helper(root, 0, lists)
	return lists

def createLinkedListsOfAllNodesAtEachDepth_helper(node, level, lists):
	if (node == None):
		return
	if (len(lists) == level):
		levelList = [node]
		lists.append(levelList)
	else:
		lists[level].append(node)
	createLinkedListsOfAllNodesAtEachDepth_helper(node.left, level + 1, lists)
	createLinkedListsOfAllNodesAtEachDepth_helper(node.right, level + 1, lists)

def inorderSuccessor(node):
	if (node.parent == None or node.right != None):
		leftChild = node.right
		while (leftChild != None and leftChild.left != None):
			leftChild = leftChild.left
		return leftChild
	else:
		q = node
		parent = node.parent
		while (parent != None and parent.left != q):
			q = parent
			parent = q.parent
		return parent

# version with Binary tree
def lowestCommonAncestor(root, node1, node2):
	if (root == None or root == node1 or root == node2):
		return root
	left = lowestCommonAncestor(root.left, node1, node2)
	right = lowestCommonAncestor(root.right, node1, node2)
	if (left != None and right != None):
		return root
	elif (left != None):
		return left
	else:
		return right

# version with Binary search tree
def lowestCommonAncestorBST(root, node1, node2):
	if (root == None or root == node1 or root == node2):
		return root
	if (max(node1.data, node2.data) < root.data):
		return lowestCommonAncestorBST(root.left, node1, node2)
	elif (min(node1.data, node2.data) > root.data):
		return lowestCommonAncestorBST(root.right, node1, node2)
	else:
		return root

def rotateMatrix(matrix):
	n = len(matrix)
	for layer in range(n / 2):
		first = layer
		last = n - 1 - layer
		for i in range(first, last):
			offset = i - first
			top = matrix[first][i]
			matrix[first][i] = matrix[last - offset][first]
			matrix[last - offset][first] = matrix[last][last - offset]
			matrix[last][last - offset] = matrix[i][last]
			matrix[i][last] = top



if __name__ == '__main__':
	main()