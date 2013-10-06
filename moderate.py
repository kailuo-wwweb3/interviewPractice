def main():
	print countFactZeros(20)


def factorsOf5(i):
	count = 0
	while (i % 5 == 0):
		count = count + 1
		i = i / 5
	return count

def countFactZeros(num):
	count = 0
	for i in range(2, num + 1):
		count = count + factorsOf5(i)
	return count
	














if __name__ == '__main__':
	main()