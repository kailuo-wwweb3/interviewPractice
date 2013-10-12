def main():
	print hasAllUniqueCharacters("abcda")
	print isPermutationEachOther("abcde", "adeab")


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

def relaceSpacesInString(string):
	charArray = list(string)
	for 


if __name__ == '__main__':
	main()