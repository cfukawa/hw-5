from collections import Counter
from matplotlib import pyplot as plt
from newsapi import NewsApiClient
from wordcloud import WordCloud
import en_core_web_lg
import pandas as pd
import pickle
import spacy
import string


def get_keywords_eng(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    result = []
    pos_tag = ['VERB', 'NOUN', 'PROPN']
    for token in doc:
        if token.text in nlp_eng.Defaults.stop_words or token.text in string.punctuation:
            continue
        elif token.pos_ in pos_tag:
            result.append(token.text)
    return result


if __name__ == '__main__':
    nlp_eng = en_core_web_lg.load()
    news_api = NewsApiClient(api_key='5ac03a80e0be450f801fcd50acf70c7d')

    articles = news_api.get_everything(q='coronavirus', language='en',
                                       from_param='2021-10-01', sort_by='relevancy')

    filename = 'articlesCOVID.pckl'
    pickle.dump(articles, open(filename, 'wb'))
    loaded_model = pickle.load(open(filename, 'rb'))
    pickle.dump(loaded_model, open(filename, 'wb'))

    data = []

    for i, article in enumerate(articles):
        for x in articles['articles']:
            title = x['title']
            date = x['publishedAt']
            description = x['description']
            content = x['content']
            data.append({'title': title, 'date': date, 'desc': description, 'content': content})

    df = pd.DataFrame(data)
    df = df.dropna()
    print(df.head(10))

    results = []

    for content in df.content.values:
        results.append([[('#' + x[0]) for x in Counter(get_keywords_eng(content)).most_common(5)]])

    df['keywords'] = results

    df.to_csv('output.csv', index=False)

    words = str(results)
    word_cloud = WordCloud(width=800, height=400, max_font_size=50, max_words=100, background_color='white')\
        .generate(words)
    plt.figure()
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis('off')
    figure = plt.gcf()
    figure.set_size_inches(12, 6)
    plt.savefig('word_cloud.png', dpi=100)
    plt.show()
