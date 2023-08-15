import logging
import sys
import os

from llama_index import SimpleDirectoryReader,VectorStoreIndex, ServiceContext
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings

from constants import (
    EMBEDDING_MODEL_NAME
)

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
    documents = SimpleDirectoryReader('data').load_data()
    embed_model = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device_type},
    )
    service_context = ServiceContext().from_defaults(embed_model=embed_model)

    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    index.storage_context.persist()

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()