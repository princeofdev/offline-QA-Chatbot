import logging
import sys
import os

import click
import torch

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from langchain.docstore.document import Document
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings

from constants import (
    DATA_PATH,
    PERSIST_PATH,
    INGEST_THREADS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    CHROMA_SETTINGS
)

def load_data(data_path: str) -> list[Document]:
    logging.info("Loading documents data ...")

    all_files = os.listdir(data_path)
    paths = []

    for file_path in all_files:
        file_extension = os.path.splitext(file_path)[1]
        data_file_path = os.path.join(data_path, file_path)
        if file_extension in DOCUMENT_MAP.keys():
            paths.append(data_file_path)

    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    chunksize = round(len(paths) / n_workers)
    docs = []

    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        for i in range(0, len(paths), chunksize):
            filepaths = paths[i : (i + chunksize)]
            future = executor.submit(load_document_batch, filepaths)
            futures.append(future)
        for future in as_completed(futures):
            contents, _ = future.result()
            docs.extend(contents)

    return docs

def load_document_batch(filepaths):
    logging.info("Loading document batch ...")

    with ThreadPoolExecutor(len(filepaths)) as exe:
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        data_list = [future.result() for future in futures]

        return (data_list, filepaths)

def load_single_document(file_path: str) -> Document:
    logging.info("Loading single document ...")

    file_extension = os.path.splitext(file_path)[1]
    loader_class = DOCUMENT_MAP.get(file_extension)

    if loader_class:
        loader = loader_class(file_path)
    else:
        raise ValueError("Document type is undefined.")

    return loader.load()[0]

def split_documents(documents: list[Document]) -> tuple[list[Document], list[Document]]:
    logging.info("Splitting documents ...")

    text_docs, python_docs = [], []

    for doc in documents:
        file_extension = os.path.splitext(doc.metadata["source"])[1]
        if file_extension == ".py":
            python_docs.append(doc)
        else:
            text_docs.append(doc)

    return text_docs, python_docs

@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
def main(device_type):
    documents=load_data(DATA_PATH)
    text_documents, python_documents = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=20)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=1024, chunk_overlap=20
    )
    texts = text_splitter.split_documents(text_documents)
    texts.extend(python_splitter.split_documents(python_documents))
    
    logging.info("Constructing embeddings ...")
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device_type},
    )

    logging.info("Creating DB ...")
    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_PATH,
        client_settings=CHROMA_SETTINGS,
    )
    db.persist()
    db = None

    logging.info("Done.")

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()