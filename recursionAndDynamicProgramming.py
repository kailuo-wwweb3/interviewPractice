import practice
import copy

def main():
	# getPathDP_main(5, 5)
	print subset([1, 2])



def countWays(n):
	if (n < 0):
		return 0
	elif (n == 0):
		return 1
	else:
		return countWays(n - 1) + countWays(n - 2) + countWays(n - 3)

# avoid repeating calculation.
def countWaysDP(n, record):
	if (n < 0):
		return 0
	elif (n == 0):
		return 1
	elif (record[n] > -1):
		return record[n]
	else:
		record[n] = countWaysDP(n - 1, record) + countWaysDP(n - 2, record) + countWaysDP(n - 3, record)
		return record[n]

def countWaysDP_main(n):
	record = []
	for i in range(n):
		record[i] = -1
	return countWaysDP(n, record)


def getPath(x, y):
	if (x == 0 and y == 0):
		return 1
	if (x < 0 or y < 0):
		return 0
	return getPath(x - 1, y) + getPath(x, y - 1)

def getPathDP(x, y, path, cache):
	if (cache.has_key((x, y))):
		return cache[(x, y)]
	path.append((x, y))
	if (x == 0 and y == 0):
		return True
	success = False
	if (x >= 1 and isFree(x - 1, y)):
		success = getPathDP(x - 1, y, path, cache)
	if (not success and y >= 1 and isFree(x, y - 1)):
		success = getPathDP(x, y - 1, path, cache)
	if (not success):
		path.remove((x, y))
	cache[(x, y)] = success
	return success

def getPathDP_main(x, y):
	path = []
	cache = {}
	getPathDP(x, y, path, cache)
	print path
def isFree(x, y):
	if (x == 1 and y == 0):
		return False
	else:
		return True


def magicIndex(array, start, end):
	if (end < start or start < 0 or end >= len(array)):
		return -1
	mid = (start + end) / 2
	if (mid == array[mid]):
		return mid
	elif (mid < array[mid]):
		return magicIndex(array, start, mid)
	else:
		return magicIndex(array, mid + 1, end)

def subset(array):
	if (len(array) == 0):
		return []
	elif (len(array) == 1):
		return [[], array]
	else:
		firstElement = array[0]
		subsets = subset(array[1:])
		subsets_copy = copy.copy(subsets)
		for element in subsets_copy:
			element_copy = copy.copy(element)
			element_copy.append(firstElement)
			subsets.append(element_copy)
	return subsets


if __name__ == '__main__':
	main()