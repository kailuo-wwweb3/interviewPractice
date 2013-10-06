import practice


class PartialSum:
	def __init__(self, sumNode, carry):
		self.sumNode = sumNode
		self.carry = carry

def main():
	a = practice.LLNode(9)
	b = practice.LLNode(2)
	c = practice.LLNode(8)
	d = practice.LLNode(4)
	a.next = b
	b.next = c
	c.next = d

	a1 = practice.LLNode(4)
	b1 = practice.LLNode(4)
	c1 = practice.LLNode(4)
	d1 = practice.LLNode(4)
	a1.next = b1
	b1.next = c1
	c1.next = d1
	# printLL(a)
	# deleteDupsWithOutBuffer(a)
	# printLL(a)

	# nthToLastIter(a, 3)
	# a = partitionLinkedList(a, 3)
	# printLL(a)
	# r = addLists1_main(a, a1)
	r = addLists2_main(a, a1)
	printLL(r)


def printLL(head):
	result = str(head.data)
	while not head.next == None:
		result = result + ("->" + str(head.next.data))
		head = head.next
	print result


def deleteDups(n):
	table = {}
	previous = None
	while (n is not None):
		if table.has_key(n.data):
			previous.next = n.next
		else:
			table[n.data] = True
			previous = n
		n = n.next

def deleteDupsWithOutBuffer(n):
	if n is None:
		return
	current = n
	while not current == None:
		runner = current
		while not runner.next == None:
			if runner.next.data == current.data:
				runner.next = runner.next.next
			else:
				runner = runner.next
		current = current.next

def nthToLast(head, k):
	if (head == None):
		return 0
	i = nthToLast(head.next, k) + 1
	if (i == k):
		print head.data
	return i

def nthToLastIter(head, k):
	if k <= 0:
		return None
	p1 = head
	p2 = head

	for i in range(k - 1):
		if p2 == None:
			return None
		p2 = p2.next

	if p2 == None:
		return None
	while (not p2.next == None):
		p1 = p1.next
		p2 = p2.next
	print p1.data

def partitionLinkedList(node, x):
	beforeStart = None
	afterStart = None

	while (not node == None):
		nextNode = node.next
		if (node.data < x):
			node.next = beforeStart
			beforeStart = node
		else:
			node.next = afterStart
			afterStart = node
		node = nextNode

	if (beforeStart == None):
		return afterStart
	printLL(beforeStart)
	printLL(afterStart)

	head = beforeStart
	while (beforeStart.next != None):
		beforeStart = beforeStart.next
	beforeStart.next = afterStart
	return head

def addLists1_main(l1, l2):
	return addLists1(l1, l2, 0)

def addLists1(l1, l2, carry):
	if l1 == None and l2 == None and carry == 0:
		return None
	result = practice.LLNode(carry)
	value = carry
	if l1 != None:
		value = value + l1.data
	if l2 != None:
		value = value + l2.data
	result.data = value % 10

	if (l1 != None or l2 != None or value >= 10):
		more = addLists1(None if l1 == None else l1.next,
			None if l2 == None else l2.next, 1 if value >= 10 else 0)
		result.next = more
	return result


# digits are stored in forward order.

def addLists2_main(l1, l2):
	sumHead = addLists2(l1, l2)
	head = sumHead.sumNode
	if (sumHead.carry > 0):
		newNode = practice.LLNode(sumHead.carry)
		newNode.next = sumHead.sumNode
		head = newNode
	return head

def addLists2(l1, l2):
	if l1 == None and l2 == None:
		return None
	value = 0
	if l1 != None:
		value = value + l1.data
	if l2 != None:
		value = value + l2.data
	previousSum = addLists2(l1.next, l2.next)
	result = practice.LLNode(value)
	if (previousSum != None):
		value = value + previousSum.carry
		result.data = value % 10
		result.next = previousSum.sumNode
	currentSum = PartialSum(result, (1 if value > 10 else 0))
	return currentSum

if __name__ == '__main__':
	main()