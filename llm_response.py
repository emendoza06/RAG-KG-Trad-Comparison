from langchain_community.graphs import Neo4jGraph
from langchain.chains import RetrievalQA
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
load_dotenv()

# Initialize your Neo4j graph connection
graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
)

#Query the graph to test its connection
result = graph.query("""
MATCH (c:Course{name: 'llm-fundamentals'})-[:HAS_MODULE]->(m:Module)
RETURN c, m
""")

print(result)

#When you connect to the Neo4J database, the object loads the database schema into memory - this enables Langchain to access the schema information
#without having to query the database
print(graph.schema)

#Connect to OpenAI to embed the queries
chat_llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

#We need to know the embedding provider in order to know how many dimensions the vectors are
embedding_proivder = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))

#Retrieve the existing paragraphs_index
paragraph_index = Neo4jVector.from_existing_index(
    embedding_proivder,
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD'),
    index_name="paragraph_index",
    embedding_node_property="embedding",
    text_node_property="text",
)

kg_retriever = RetrievalQA.from_llm(
    llm=chat_llm,
    retriever=paragraph_index.as_retriever(),
    verbose=True,
    return_source_documents=True
)

result = kg_retriever.invoke(
    {"query": "Who is Michael Scott from The Office?"}
)

print(result)



