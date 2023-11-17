from langchain.text_splitter import CharacterTextSplitter


def calculate_embeddings(text):
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=5000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    documents = text_splitter.create_documents([text])
    return documents
