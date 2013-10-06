import sys
def main():
	# f = open(sys.argv[1], 'r')
	# o = open("output.txt", 'w')
	print KMP_search("adfasdfasdfafd", "asd")



def KMP_search(text, search_string):
	indexLs = []

	next = preComputeTable(search_string)
	j = 0
	for i in range(len(text)):
		if (text[i] == search_string[j]):
			j += 1
		else:
			j = next[j]
		if (j == len(search_string)):
			indexLs.append(i - len(search_string) + 1)
	return indexLs


def preComputeTable(search_string):
	next = {}
	for i in range(len(search_string)):
		next[i] = 0
	x = 0
	for j in range(len(search_string)):
		if (search_string[x] == search_string[j]):
			next[j] = next[x]
			x += 1
		else:
			next[j] = x + 1
			x = next[x]
	return next



if __name__ == '__main__':
	main()