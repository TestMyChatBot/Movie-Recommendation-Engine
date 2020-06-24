import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

cv = CountVectorizer()
text = ["London Paris London","Paris Paris London"]

cv_fit = cv.fit_transform(text)

#print("Frequency of Words\n",cv_fit.toarray())

cs = cosine_similarity(cv_fit)
#print("Cosin Similarity of Words\n",cs)

movie_df = pd.read_csv("movie_dataset.csv")

print(movie_df.head())