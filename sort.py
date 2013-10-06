import random
def main():
	# array = [9, 6, 7, 8, 2, 3, 1, 0]
	array = [3,1,2,5,4]
	# print array
	countingSort(array, 5)
	print array

def bubbleSort(array):
	n = len(array)
	while True:
		swapped = False
		for i in range(1, n):
			if array[i - 1] > array[i]:
				temp = array[i]
				array[i] = array[i - 1]
				array[i - 1] = temp
				swapped = True
		if not swapped:
			break

def selectionSort(array):
	n = len(array)
	for j in range(n - 1):
		iMin = j
		for i in range(j + 1, n):
			if array[i] < array[iMin]:
				iMin = i
		if iMin != j:
			temp = array[j]
			array[j] = array[iMin]
			array[iMin] = temp

def insertionSort(array):
	n = len(array)
	for i in range(1, n):
		item = array[i]
		iHole = i
		while iHole > 0 and array[iHole - 1] > item:
			array[iHole] = array[iHole - 1]
			iHole = iHole - 1
		array[iHole] = item

def merge_sort(array):
	if len(array) <= 1:
		return array
	middle = int(len(array) / 2)
	left = []
	right = []
	for x in range(0, middle):
		left.append(array[x])
	for x in range(middle, len(array)):
		right.append(array[x])
	left = merge_sort(left)
	right = merge_sort(right)
	return merge(left, right)

def merge(left, right):
	result = []
	while len(left) > 0 or len(right) > 0:
		if len(left) > 0 and len(right) > 0:
			if left[0] <= right[0]:
				result.append(left[0])
				left = left[1:len(left)]
			else:
				result.append(right[0])
				right = right[1:len(right)]
		elif len(left) > 0:
			result.append(left[0])
			left = left[1:len(left)]
		elif len(right) > 0:
			result.append(right[0])
			right = right[1:len(right)]
	return result

def quickSort(array):
	random.shuffle(array)
	print array
	quickSort_helper(array, 0, len(array) - 1)
	print array	
	
def quickSort_helper(array, lo, hi):
		if hi <= lo:
			return
		j = partition(array, lo, hi)
		quickSort_helper(array, lo, j - 1)
		quickSort_helper(array, j + 1, hi)

def partition(array, lo, hi):
	i = lo + 1
	j = hi
	
	print array
	while True:
		while array[i] < array[lo]:
			if i == hi:
				break
			i = i + 1
		while array[lo] < array[j]:
			if j == lo:
				break
			j = j - 1
		if i >= j:
			break
		array[i], array[j] = array[j], array[i]
	array[lo], array[j] = array[j], array[lo]
	return j
def quickSelect(array, k):
	random.shuffle(array)
	lo = 0
	hi = len(array) - 1
	while hi > lo:
		j = partition(array, lo, hi)
		if j < k:
			lo = j + 1
		elif j > k:
			hi = j - 1
		else:
			return array[k]
	return array[k]


def countingSort(array, maxval):
	m = maxval + 1
	count = [0] * m
	for a in array:
		count[a] += 1
	print count
	i = 0
	for a in range(m):
		for c in range(count[a]):
			array[i] = a
			i += 1
	return array



if __name__ == '__main__':
	main()
