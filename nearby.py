import sys
import math
import heapq

class TopicDisRecord(object):
	"""docstring fotopicDisRecordme"""
	def __init__(self, topicID, distance):
		self.topicID = topicID
		self.distance = distance

	def __cmp__(self, obj):
		if (self.distance != obj.distance):
			return self.distance - obj.distance
		else:
			return self.topicID - obj.topicID

def main():
	f = open(sys.argv[1], 'r')
	o = open("output.txt", 'w')

	firstLine = f.readline()[:-1].split(" ")
	T = int(firstLine[0])
	Q = int(firstLine[1])
	N = int(firstLine[2])

	topics = {}
	for i in range(T):
		topic_record = f.readline()[:-1].split(" ")
		topics[topic_record[0]] = (topic_record[1], topic_record[2])
	questions = []
	for i in range(Q):
		questions.append(f.readline()[:-1].split(" "))
	results = []
	for i in range(N):
		if (i == N - 1):
			results.append(f.readline().split(" "))
		else:
			results.append(f.readline()[:-1].split(" "))
	for i in results:
		if (i[0] == "t"):
			c = i[1]
			location = tuple(i[2:])
			heap = fetchNearestNTopics(c, topics, location)
			for i in range(len(heap)):
				if (i == len(heap) - 1):
					o.write(heap[i].topicID + "\n")
				else:
					o.write(heap[i].topicID + " ")
		


def fetchNearestNTopics(n, topics, location):
	heap = []
	for t in topics:
		record = TopicDisRecord(t[0], distanceBetweenTwoLocations(location, tuple(topics[t])))
		heapq.heappush(heap, record)
		if (len(heap) == int(n) + 1):
			heap.pop()
	return heap


def distanceBetweenTwoLocations(l1, l2):
	return math.sqrt(math.pow(float(l1[0]) - float(l2[0]), 2) + math.pow(float(l1[1]) - float(l2[1]), 2))




if __name__ == '__main__':
	main()