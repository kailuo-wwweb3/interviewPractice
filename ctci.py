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

	head1 = convertArrayToLinkedList([7,1,6])
	head2 = convertArrayToLinkedList([5,9,2])
	# google.displayLL(head)
	# google.displayLL(removeDuplicatesFromUnsortedLinkedListWithoutBuffer(head))
	# google.displayLL(partitionLinkedList(head, 3))
	google.displayLL(addTwoNumbers(head1, head2))


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













if __name__ == '__main__':
	main()