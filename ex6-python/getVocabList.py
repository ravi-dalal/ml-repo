def getVocabList():
	fid = open('data/vocab.txt', 'r')
	n = 1899
	vocabList = {}
	for line in fid.readlines():
		i, word = line.split()
		#print(i, word)
		vocabList[word] = int(i)
	fid.close()
	return vocabList