import google
import practice
def main():
	# n1 = practice.LLNode(1)
	# n2 = practice.LLNode(2)
	# n3 = practice.LLNode(3)
	# n4 = practice.LLNode(4)
	# n5 = practice.LLNode(5)
	# n6 = practice.LLNode(6)
	# n7 = practice.LLNode(7)
	# n1.next = n2
	# n2.next = n3
	# n3.next = n4
	# n4.next = n5
	# n5.next = n6
	# n6.next = n7
	# head = removeNthNodeFromEndOfList(n1, 4)
	# google.displayLL(head)

	mapping = {'2':['a', 'b', 'c'], '3':['d', 'e', 'f']}
	print letterCombinationsOfAPhoneNumber("23322", mapping)

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
	if (array == []):
		return []
	if (len(array) == 1 and (target / ))





if __name__ == '__main__':
	main()