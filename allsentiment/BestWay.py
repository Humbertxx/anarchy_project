from sentence_transformers import CrossEncoder
import spacy
from processingFile import sql_processing
from config import DATA_DIR
from collections import defaultdict

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

sentence_tokenizer = spacy.load("en_core_web_sm")

def main():
    rows= sql_processing(DATA_DIR)
    rows = group_chunk(rows)
        
    for _, txt_list in rows.items():
        full_text = " ".join(txt_list)
        sent = get_sentences(full_text)
        chunks = list(sentence_windows(sent, min_size=1, max_size=3))
    
    query = "most relevant quote?"
    scores = model.predict([(query, passage) for passage in chunks])
    print(scores)
    
    print ("NOW RANKS"+ ("-"* 9))
    ranks = model.rank(query, chunks)
    
    print(ranks)
    print("Query:", query)
    for rank in ranks:
        print(f"{rank['score']:.2f}\t{chunks[rank['corpus_id']]}")

def group_chunk(rows):
    texts_by_id = defaultdict(list)
    for id, txt in rows:
        texts_by_id[id].append(txt)
    return texts_by_id
    
def get_sentences(text):
    doc = sentence_tokenizer(text)
    return [sent.text.strip() for sent in doc.sents]
    
def sentence_windows(sentence, min_size = 1, max_size= 3):
    n = len(sentence)
    for start in range(n):
        for size in range(min_size, max_size + 1):
            end = start + size
            if end <= n:
                yield " ".join(sentence[start:end])
	

if __name__ == "__main__":
    main()
