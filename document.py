import os
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, DirectoryLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_all_pdf(pdf_directory):
    all_documents = []
    pdf_dir = Path(pdf_directory)

    pdf_files = list(pdf_dir.glob('**/*.pdf'))

    print(f"Found{len(pdf_files)} PDF files")

    for pdf in pdf_files:
        print(f"\nProcessing: {pdf.name}")
        try:
            loader = PyPDFLoader(str(pdf))
            documents = loader.load()

            for doc in documents:
                doc.metadata['source_file'] = pdf.name
                doc.metadata['file_type'] = 'pdf'

            all_documents.extend(documents)
            print(f"✔ Loaded: {len(documents)} pages")

        except Exception as e:
            print(f"Error {e}")

    print(f"\n Total Documents Loaded: {len(all_documents)}")
    return all_documents

all_pdf_documents = process_all_pdf("./pdf")


def split_documents(documents,chunk_size=1000,chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n","\n"," ",","]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"\nSplit Documents: {len(documents)} documents into {len(split_docs)} chunks")

    if split_docs:
        print(f"\n Example Chunk")
        print(f"Content: {split_docs[0].page_content[:200]}...")
        print(f"Meta Data {split_docs[0].metadata}")

    return split_docs



chunks = split_documents(all_pdf_documents)
print(chunks)