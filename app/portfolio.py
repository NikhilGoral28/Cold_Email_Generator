import pandas as pd
import chromadb
import uuid
from chromadb.utils import embedding_functions
from pathlib import Path

class Portfolio:

    def __init__(self, file_path=None):
        if file_path is None:
            # app/portfolio.py → project root → resource/my_portfolio.csv
            base_dir = Path(__file__).resolve().parent.parent
            file_path = base_dir / "resource" / "my_portfolio.csv"


        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.chroma_client = chromadb.PersistentClient("vectorstore")
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction()
        self.collection = self.chroma_client.create_collection(name="Portfolio",embedding_function=sentence_transformer_ef,get_or_create=True)


    def load_portfolio(self):
        if not self.collection.count():
            for _, row in self.data.iterrows():
                self.collection.add(
                documents=[row["Techstack"]],
                metadatas=[{"links": row["Links"]}],
                ids=[str(uuid.uuid4())]
                )
                
    
    def query_links(self, skills):
        return self.collection.query(query_texts=skills, n_results=2).get("metadatas",[])
        
        