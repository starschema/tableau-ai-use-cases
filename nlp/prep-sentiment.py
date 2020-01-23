from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tabpy.models.utils import setup_utils


import ssl
_ctx = ssl._create_unverified_context
ssl._create_default_https_context = _ctx

# nltk.download('vader_lexicon')
# nltk.download('punkt')


def lookup(df):
    scores = []

    result = pd.DataFrame(columns=['sentiment_score', 'F1'])
    sid = SentimentIntensityAnalyzer()
    for text in df['Title'].tolist():
        sentimentResults = sid.polarity_scores(text)
        score = sentimentResults['compound']
        scores.append(score)

    result['sentiment_score'] = scores
    result['F1'] = df['F1']

    return result


def get_output_schema():
    return pd.DataFrame({
        'sentiment_score': prep_decimal(),
        'F1': prep_int()
    })
