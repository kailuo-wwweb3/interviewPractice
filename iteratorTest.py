


def main():
	a = chain("abc", "ddd")
	for i in a:
		print i

def chain(*iterables):
	for it in iterables:
		for element in it:
			yield element










if __name__ == '__main__':
	main()