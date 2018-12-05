import jieba
import jieba.analyse
from gensim.models.word2vec import LineSentence
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

def docvec_function():
	documents = []
	count = 0
	f = open('/Users/Nick/Desktop/未命名文件夹/learning-nlp/chapter-7/word2vec训练与相似度计算/data/P1.txt','r')
	for line in f:
		words_line = jieba.cut(line)
		words_line = list(words_line)
		documents.append(TaggedDocument(words_line, [str(count)]))
		count+=1
	model = Doc2Vec(documents, dm = 1, size = 100, window = 8, min_count = 5)
	print(model.docvecs.most_similar('0'))
	print(model.docvecs.similarity('0','1'))
	print(model.docvecs['10'])
	words = "姚明明天开始比赛"
	words = jieba.cut(words)
	print(model.infer_vector(list(words)))
	print(model['姚明'])

if __name__ == '__main__':
	docvec_function()