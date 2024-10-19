import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import OpenAIEmbeddings


COURSES_PATH = "1-knowledge-graphs-vectors/data/asciidoc"

# Load lesson documents
loader = DirectoryLoader(COURSES_PATH, glob="**/lesson.adoc", loader_cls=TextLoader)
docs = loader.load()

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1500,
    chunk_overlap=200,
)

chunks = text_splitter.split_documents(docs)

print(chunks)

graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD'),
)

neo4j_vector = Neo4jVector.from_documents(
    chunks,
    OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY')),
    graph=graph,
    index_name="chunkVector",
    node_label="Chunk", 
    text_node_property="text", 
    embedding_node_property="embedding",  
)