import sys
def main():
	f = open(sys.argv[1], 'r')
	o = open("output.txt", 'w')
	for i in f.readlines():
		parsed = i.split("|")
		string = parsed[0]
		number = parsed[1]
		ls = splitString(string, int(number))
		if i[-1] == "\n":
			if isPalidrome(ls):
				o.write(i[:-1] + "|1" + "\n")
			else:
				o.write(i[:-1] + "|0" + "\n")
		else:
			if isPalidrome(ls):
				o.write(i + "|1" + "\n")
			else:
				o.write(i + "|0" + "\n")


def splitString(string, number):
	result = []
	for i in range(0, len(string), number):
		result.append(string[i:i + number])
	return result


def isPalidrome(ls):
	result = []
	for i in reversed(ls):
		result.append(i)
	return result == ls
if __name__ == '__main__':
	main()