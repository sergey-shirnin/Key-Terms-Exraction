from nltk.tokenize import word_tokenize
from collections import Counter
from lxml import etree


xml_file = "news.xml"
root = etree.parse(xml_file).getroot()
news = root[0]

heads = [el.text for n in news for el in n if el.get('name') == 'head']
texts = [el.text for n in news for el in n if el.get('name') == 'text']

for head, text in zip(heads, texts):
    tokens = sorted(word_tokenize(text.lower()), reverse=True)
    most_common = [w for w, _ in Counter(tokens).most_common(5)]
    print(''.join((head, ':')))
    print(' '.join(most_common))
