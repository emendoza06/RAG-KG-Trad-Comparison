import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from openai import OpenAI
import numpy as np
import json

COURSES_PATH = "data/asciidoc"

loader = DirectoryLoader(COURSES_PATH, glob="**/lesson.adoc", loader_cls=TextLoader)
docs = loader.load()

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1500,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(docs)

#Collect all embeddings 
embeddings = []
#Collect all plain text
texts = []

llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

#Def for getting embeddings from llm
def get_embedding(llm, chunk):
    response = llm.embeddings.create(
        input=chunk.page_content,
        model="text-embedding-ada-002"
    )
    return np.array(response.data[0].embedding)

#Create embeddings array and build tree
for chunk in chunks:
    embedding = get_embedding(llm, chunk)
    embeddings.append(embedding)
    texts.append(chunk.page_content)

np.save('embeddings.npy', np.array(embeddings))
with open('paragraph_texts.json', 'w') as f:
    json.dump(texts, f)

