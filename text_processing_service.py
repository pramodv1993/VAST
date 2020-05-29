import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class TextProcessing:
	def __init__(self):
		self.lemmatizer = WordNetLemmatizer()
		self.stop_words = set(stopwords.words('english'))

	def remove_punct(self, words):
	    return [word for word in words if word.isalpha()]

	def remove_stopwords(self, words):
	    return [word for word in words if word not in self.stop_words]  

	def lemmatize(self, words):
	    return [self.lemmatizer.lemmatize(word) for word in words]

	def process_sentence(self, sentence):
	    words = word_tokenize(sentence)
	    words = self.remove_punct(words)
	    words = [word.lower() for word in words]
	    words = self.remove_stopwords(words)
	    words = self.lemmatize(words)
	    return words