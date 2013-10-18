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
	print NQueens()

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

GRIDSIZE = 4
def NQueens():
	result = []
	columns = [0] * GRIDSIZE
	placeQueens(0, columns, result)
	return result

def placeQueens(row, columns, result):
	if (row == GRIDSIZE):
		result.append(copy.copy(columns))
		return
	for column in range(GRIDSIZE):
		if (checkValid(row, column, columns)):
			columns[column] = row
			placeQueens(row + 1, columns, result)

def checkValid(row, column, columns):
	for column2 in range(len(columns)):
		if (columns[column2] == row):
			return False
		if (abs(columns[column2] - row) == abs(column - column2)):
			return False
	return True




if __name__ == '__main__':
	main()