from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from lxml import etree
from string import punctuation
from collections import Counter
import nltk


xml_file = "news.xml"

news_list = (news for news in etree.parse(xml_file).getroot()[0])

filter_items = set(punctuation) | set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer().lemmatize

n = 0
while n < 10:
    news = next(news_list)
    head = news.find("value[@name='head']")
    text = news.find("value[@name='text']")
    tokens = (ntoken for token in word_tokenize(text.text.lower())
              if (ntoken := lemmatizer(token)) not in filter_items)
    most_common = []
    counter = Counter(sorted(tokens, reverse=True)).most_common()
    i = 0
    for w, _ in counter:
        if i == 5:
            break
        if nltk.pos_tag([w])[0][1] == 'NN':
            most_common.append(w)
            i += 1
    n += 1
    print(''.join((head.text, ':')))
    print(*most_common, end='\n\n')
