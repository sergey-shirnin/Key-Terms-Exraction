from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from lxml import etree
from string import punctuation
import nltk


class KeyTermsExtractor:
    def __init__(self, file: str, pos: str):
        self.file = file
        self.pos = pos
        self.filter_items = set(punctuation) | set(stopwords.words('english'))
        self.corpus = None
        self.heads = []
        self.dataset = []
        self.terms = []
        self.lemmatizer = WordNetLemmatizer().lemmatize
        self.vectorizer = TfidfVectorizer()

    def get_corpus(self):
        self.corpus = [text for text in etree.parse(self.file).getroot()[0]]

    def get_dataset(self):
        for text in self.corpus:
            head, content = [text.find(f'value[@name="{what}"]') for what in ("head", "text")]
            tokens = [ntoken for token in word_tokenize(content.text.lower())
                      if (ntoken := self.lemmatizer(token)) not in self.filter_items and \
                      nltk.pos_tag([ntoken])[0][1] == self.pos]
            self.dataset.append(' '.join(tokens))
            self.heads.append(head)

    def get_terms(self):
        tfidf_matrix = self.vectorizer.fit_transform(self.dataset)
        terms = self.vectorizer.get_feature_names_out()
        for i in range(len(self.heads)):
            words_score = ((term, score) for term, score in zip(terms, tfidf_matrix.toarray()[i]))
            words_score_sorted = sorted(words_score, key=lambda item: (item[1], item[0]), reverse=True)
            self.terms.append(' '.join(w_s[0] for w_s in words_score_sorted[:5]))

    def main(self):
        self.get_corpus()
        self.get_dataset()
        self.get_terms()
        for head, terms in zip(self.heads, self.terms):
            print(''.join((head.text, ':')))
            print(terms, end='\n\n')


my_extract = KeyTermsExtractor(file="news.xml", pos='NN')
my_extract.main()
