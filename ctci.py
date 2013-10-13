def main():
	print hasAllUniqueCharacters("abcda")
	print isPermutationEachOther("abcde", "adeab")
	test1 = list("Mr John Smith    ")
	relaceSpacesInString(test1)
	print test1
	print compressString("aabcccccaaa")



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

# def rotateMatrix(matrix):
	

if __name__ == '__main__':
	main()