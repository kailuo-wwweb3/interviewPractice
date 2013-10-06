import practice
import math
from collections import defaultdict
from collections import deque


def main():
	# root = practice.Node(1)
	# left1 = practice.Node(1)
	# left2 = practice.Node(1)
	# root.left = left1
	# left1.right = left2
	# print isBalanced(root)
	root = practice.convertSortedArrayToBST([0,1,2,3,4,5,6,7,8,9,10])
	# lists = createLevelList_main(root)
	# for i in lists:
	# 	result = []
	# 	for node in i:
	# 		result.append(node.data)
	# 	print result
	# findSum_main(root, 7)
	# inorderTraversal(root)
	# preorderTraversal(root)
	levelorderTraversal(root)



def isBalanced(root):
	if (checkHeight(root) == -1):
		return False
	else:
		return True

def checkHeight(node):
	if (node == None):
		return 0

	# check if left is balanced.
	leftHeight = checkHeight(node.left)
	if (leftHeight == -1):
		return -1

	rightHeight = checkHeight(node.right)
	if (rightHeight == -1):
		return -1
	heightDiff = abs(leftHeight - rightHeight)
	if (heightDiff > 1):
		return -1
	else:
		return max(leftHeight, rightHeight) + 1

def createLevelList(node, lists, level):
	if (node == None):
		return None

	newList = []
	if (len(lists) == level):
		lists.append(newList)
	else:
		newList = lists[level]
	newList.append(node)
	createLevelList(node.left, lists, level + 1)
	createLevelList(node.right, lists, level + 1)

def createLevelList_main(root):
	lists = []
	createLevelList(root, lists, 0)
	return lists

def checkBST_main(root):
	return checkBST(root, float("inf"), float("-inf"))

def checkBST(node, maxNumber, minNumber):
	if (node == None):
		return True
	if (node.data <= minNumber or node.data >= maxNumber):
		return False
	else:
		return checkBST(node.left, node.data, minNumber) and checkBST(node.right, maxNumber, node.data)

def findSum(node, sumVal, path, level):
	if (node == None):
		return
	if (len(path) != level + 1):
		path.append(node.data)
	else:
		path[level] = node.data

	t = 0
	# print (path, level)
	for i in range(level, -1, -1):
		t = t + path[i]
		if (t == sumVal):
			print (path, i, level)	
	findSum(node.left, sumVal, path, level + 1)
	findSum(node.right, sumVal, path, level + 1)

	path.remove(path[len(path) - 1])

def findSum_main(root, sumVal):
	findSum(root, sumVal, [], 0)

def depth(root):
	if (root == None):
		return 0
	return 1 + max(depth(root.left), depth(root.right))

def inorderTraversal(root):
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

def preorderTraversal(root):
	stack = []
	node = root
	while (stack != [] or node != None):
		if node != None:
			print node.data
			if (node.right != None):
				stack.append(node.right)
			node = node.left
		else:
			node = stack.pop()

def levelorderTraversal(root):
	queue = deque([])
	queue.append(root)
	while (len(queue) > 0):
		current = queue.popleft()
		print current.data
		if (current.left != None):
			queue.append(current.left)
		if (current.right != None):
			queue.append(current.right)





 
 #shi shi da shuai shuai!!~~~ <3 <3 hei hei hei 

if __name__ == '__main__':
	main()