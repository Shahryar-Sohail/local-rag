#from langchain_core.documents import Document

#doc = Document(
 #   page_content="This is a test page",
  #  metadata={
   #     "author": "Shahryar",
    #    "title": "Test Title",
     #   "description": "This is a test description",
    #}
#)

#print(doc.metadata["author"])

#----------------------------------
# from langchain_community.document_loaders import TextLoader
# loader = TextLoader("test.txt",encoding="utf-8")
# doc = loader.load()
# print(doc)
#-----------------------------------
#
# from langchain_community.document_loaders import DirectoryLoader, TextLoader
#
# dir_loader = DirectoryLoader(
#     "./files",
#     glob="**/*.txt",
#     loader_cls=TextLoader,
#     loader_kwargs={'encoding': 'utf-8'},
# )
# documents = dir_loader.load()
# print(documents)
#------------------------------------
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, DirectoryLoader
dir_loader = DirectoryLoader(
    "./pdf",
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader,
)
documents = dir_loader.load()
# print(documents)
print(type(documents[0]))

