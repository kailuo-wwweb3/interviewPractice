def main():
	# print mergeTwoSortedArray([1,2,3,4,5], [6,7,8,9,10])
	# print sortToMakeAnagramTogether(["abc", "bdaaadfa", "bca", "mnb", "bmn"])
	print searchR(["a", "", "b", "c"], "b", 0, 2)


def stringCompress(s):
	result = ""
	currentChar = s[0]
	count = 1
	for i in range(1, len(s)):
		if s[i] == currentChar:
			count = count + 1
		else:
			result = result + (currentChar + str(count))
			count = 1
			currentChar = s[i]
	result = result + (currentChar + str(count))
	if len(result) >= len(s):
		return s
	else:
		return result

def rotateImageBy90(matrix):
	n = len(matrix)
	for layer in range(n/2):
		first = layer
		last = n - 1 - layer
		for i in range(first, last):
			offset = i - first

			top = matrix[first][i]
			matrix[first][i] = matrix[last - offset][first]
			matrix[last - offset][first] = matrix[last][last - offset]
			matrix[last][last - offset] = matrix[i][last]
			matrix[i][last] = top
	print matrix
	return

def ifAnElementIs0ThenSetEntireRowAndColumnToBe0(matrix):
	row = []
	column = []
	for i in range(len(matrix)):
		row.append(False)
	for i in range(len(matrix[0])):
		column.append(False)

	for i in range(len(matrix)):
		for j in range(len(matrix[0])):
			if matrix[i][j] == 0:
				row[i] = True
				column[j] = True
	for i in range(len(matrix)):
		for j in range(len(matrix[0])):
			if (row[i] and column[j]):
				matrix[i][j] = 0
	print matrix

def mergeTwoSortedArray(a1, a2):
	result = []
	index1 = 0
	index2 = 0
	while index1 < len(a1) and index2 < len(a2):
		if (a1[index1] < a2[index2]):
			result.append(a1[index1])
			index1 = index1 + 1
		else:
			result.append(a2[index2])
			index2 = index2 + 1
	if index1 < len(a1):
		for i in range(index1, len(a1)):
			result.append(a1[i])

	if index2 < len(a2):
		for i in range(index2, len(a2)):
			result.append(a2[i])
	return result

def sortToMakeAnagramTogether(array):
	hashTable = {}
	result = []
	for s in array:
		sortedString = sortString(s)
		if (not hashTable.has_key(sortedString)):
			hashTable[sortedString] = []
		hashTable[sortedString].append(s)
	for key in hashTable.keys():
		result = result + hashTable[key]
	return result

def sortString(s):
	return ''.join(sorted(s))

def searchR(array, string, start, end):
	mid = (start + end) / 2
	if (array[mid] == ""):
		first = mid - 1
		last = mid + 1
		while (True):
			if (first < start and last > end):
				return -1
			elif (start <= first and array[first] != ""):
				mid = first
				break
			elif (last <= end and array[last] != ""):
				mid = last
				break
			first = first - 1
			last = last + 1
	if (array[mid] == string):
		return mid
	elif (array[mid] < string):
		return searchR(array, string, mid + 1, end)
	else:
		return searchR(array, string, start, mid - 1) 



if __name__ == '__main__':
	main()