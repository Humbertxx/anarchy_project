from processingFile import sql_processing
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict

topic_model = BERTopic(verbose=True)
model = SentenceTransformer("all-MiniLM-L6-v2")
doc_topic_count = defaultdict(lambda: defaultdict(int))


## gather from processing in sql all texts and their ids
def get_topics(file_dir):    
    txt = sql_processing(file_dir)
    #txt = group_chunk(rows)
    embedded = model.encode(txt, show_progress_bar=True)
    topics, probs = topic_model.fit_transform(txt, embedded)
    
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 3), min_df=10)
    topic_model.update_topics(txt, vectorizer_model=vectorizer_model)
    
    return topics, probs