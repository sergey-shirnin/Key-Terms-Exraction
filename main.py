from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from lxml import etree
from string import punctuation
from collections import Counter

xml_file = "news.xml"

root = etree.parse(xml_file).getroot()
news = root[0]

heads = [el.text for n in news for el in n if el.get('name') == 'head']
texts = (el.text for n in news for el in n if el.get('name') == 'text')

filter_items = list(punctuation) + stopwords.words('english')
lemmatiser = WordNetLemmatizer()

for head in heads:
    text = next(texts)
    tokens = word_tokenize(text.lower())
    tokens_lemma = [lemmatiser.lemmatize(w) for w in tokens]
    tokens_lemma_filtered = list(filter(lambda w: w not in filter_items, tokens_lemma))
    tokens_lemma_filtered.sort(reverse=True)

    counter = Counter(tokens_lemma_filtered)
    most_common = [w for w, _ in counter.most_common(5)]

    print(''.join((head, ':')))
    print(' '.join(most_common))
