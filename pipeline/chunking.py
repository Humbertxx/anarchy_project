from langchain_text_splitters import RecursiveCharacterTextSplitter


## Text chunk, chunk size, separators, and overlap is define in the chunk, returns a list
def to_chunks(txt : str) -> list[str]:
    text = str(txt)
    text_splitter = RecursiveCharacterTextSplitter(
        separators =[" ", ""],
        chunk_size = 800,
        chunk_overlap = 160,
        )
    return text_splitter.split_text(text)