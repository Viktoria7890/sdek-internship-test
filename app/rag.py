import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from app.config import settings


def get_embeddings() -> Embeddings:
    if settings.embedding_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embedding_model,
        )
    else:
        from langchain_openai import OpenAIEmbeddings
        kwargs = {"model": settings.openai_embedding_model}
        if settings.openai_api_key:
            kwargs["api_key"] = settings.openai_api_key
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        return OpenAIEmbeddings(**kwargs)


def build_vectorstore() -> FAISS:
    embeddings = get_embeddings()
    documents = []

    for filename in sorted(os.listdir(settings.data_dir)):
        if filename.endswith(".txt"):
            filepath = os.path.join(settings.data_dir, filename)
            loader = TextLoader(filepath, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename
            documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    splits = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore
