import getpass
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv 
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders.csv_loader import UnstructuredCSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import xgboost as xgb

from langchain.document_loaders import DataFrameLoader


load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

# Load CSV file
df = pd.read_csv("data.csv", sep="\t")

# Preprocess data


# Create the descriptions for each disease to store in the vector store
df['content'] = df.apply(lambda row: f"Bệnh: {row['disease']}. Triệu chứng: {', '.join(row['symptoms_list'])}", axis=1)

# Create the document objects to store in the vector store
loader = DataFrameLoader(df, page_content_column="content")
documents = loader.load()

embeddings = OpenAIEmbeddings()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_documents(documents)

vectorstore = Chroma.from_documents(documents=split_docs, embedding=embeddings)

# 4. Chuẩn bị mô hình XGBoost
# Tạo one-hot encoding cho triệu chứng
all_symptoms = set()
for symptom_list in df['symptoms_list']:
    all_symptoms.update(symptom_list)

X = pd.DataFrame(columns=list(all_symptoms))
for idx, row in df.iterrows():
    for symptom in row['symptoms_list']:
        X.loc[idx, symptom] = 1
        
X = X.fillna(0)
y = df['disease']

# Huấn luyện XGBoost
model = xgb.XGBClassifier(n_estimators=100, random_state=42)
model.fit(X, y)