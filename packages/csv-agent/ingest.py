from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()
loader = CSVLoader("/Users/kumrohik/Downloads/Coding/Projects/genaiagent/data.csv")

docs = loader.load()
index_creator = VectorstoreIndexCreator(vectorstore_cls=FAISS )

index = index_creator.from_documents(docs)

index.vectorstore.save_local("leadtime_data")
