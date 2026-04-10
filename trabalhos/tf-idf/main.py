import os
import string
import pandas as pd
import kagglehub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

path = kagglehub.dataset_download("jrobischon/wikipedia-movie-plots")
arquivo_csv = os.path.join(path, "wiki_movie_plots_deduped.csv")

df = pd.read_csv(arquivo_csv)

documents = df['Plot'].dropna().astype(str).tolist()

query = ["alien mars spaceship"]

stopwords_en = [
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", 
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", 
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", 
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", 
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", 
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", 
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", 
    "at", "by", "for", "with", "about", "against", "between", "into", "through", 
    "during", "before", "after", "above", "below", "to", "from", "up", "down", 
    "in", "out", "on", "off", "over", "under", "again", "further", "then", 
    "once", "here", "there", "when", "where", "why", "how", "all", "any", 
    "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", 
    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", 
    "will", "just", "don", "should", "now"
]

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords_en]
    return " ".join(tokens)

docs_preprocessed = [preprocess_text(doc) for doc in documents]
query_preprocessed = [preprocess_text(query[0])]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(docs_preprocessed)
query_vector = vectorizer.transform(query_preprocessed)

cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

results_df = pd.DataFrame({
    "Title": df['Title'],
    "Plot": documents,
    "Score_TFIDF": cosine_similarities
})

results_df = results_df.sort_values(by="Score_TFIDF", ascending=False).reset_index(drop=True)

print(results_df[['Score_TFIDF', 'Title']].head(10))