import streamlit as st
import pandas as pd
import re
import os
from openai import OpenAI
import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
import nltk

nltk.download("punkt")

# Initialize the local LLM client
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Recursive Text Splitter
def recursive_text_splitter(text, max_chunk_length=1000, overlap=100):
    """
    Helper function for chunking text recursively
    """
    result = []
    current_chunk_count = 0
    separator = ["\n", " "]
    _splits = re.split(f"({'|'.join(separator)})", text)
    splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]

    for i in range(len(splits)):
        if current_chunk_count != 0:
            chunk = "".join(
                splits[
                    current_chunk_count - overlap : current_chunk_count + max_chunk_length
                ]
            )
        else:
            chunk = "".join(splits[0:max_chunk_length])

        if len(chunk) > 0:
            result.append("".join(chunk))
        current_chunk_count += max_chunk_length

    return result

# Define schema for table with embedding api
model = get_registry().get("colbert").create(name="colbert-ir/colbertv2.0")

class TextModel(LanceModel):
    text: str = model.SourceField()
    vector: Vector(model.ndims()) = model.VectorField()

# Add in vector db
def lanceDBConnection(df):
    db = lancedb.connect("/tmp/lancedb")
    table = db.create_table(
        "policies",
        schema=TextModel,
        mode="overwrite",
    )
    table.add(df)
    return table

# Read all markdown files from the directory
def read_policies_from_directory(directory):
    all_chunks = []
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                text_data = file.read()
                chunks = recursive_text_splitter(text_data, max_chunk_length=100, overlap=10)
                all_chunks.extend(chunks)
    return all_chunks

# Initialize the Streamlit app
st.title("Company Policies Q&A using Local LLM")
st.caption("Ask questions based on preloaded company policies stored in markdown format.")

# Read and process policies from the specified directory
policy_directory = "./policies"  # Update with the actual path to your directory
policy_chunks = read_policies_from_directory(policy_directory)
df = pd.DataFrame({"text": policy_chunks})
table = lanceDBConnection(df)

st.write("Policies successfully loaded and stored in the vector database.")

# Accept user question
question = st.text_input("Ask a question about the company policies:")
if question:
    # Semantic Search
    result = table.search(question).limit(5).to_list()
    context = [r["text"] for r in result]

    # Context Prompt
    base_prompt = """You are a virtual assistant specialized in understanding and explaining company policies. Your task is to read the user question, consult the provided contexts, and generate an accurate, detailed, and clear response based on the policies. Every answer you provide should include specific citations from the provided contexts in the format "Answer [position]", for example: "Employees are entitled to two weeks of paid vacation per year [1][2]." If the provided context does not contain the answer, simply state, "The provided context does not contain the answer."

User question: {}

Contexts:
{}
"""
    prompt = base_prompt.format(question, context)

    # Generate response using local LLM
    response = client.chat.completions.create(
         model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf", 
        messages=[{"role": "system", "content": prompt}],
        temperature=0.7
    )

    # Display the response
    st.write("Response from LLM:")
    st.markdown(response.choices[0].message.content, unsafe_allow_html=True)
