import streamlit as st
import pandas as pd
import re
import os
import yaml
from yaml.loader import SafeLoader
from openai import OpenAI
import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
import streamlit_authenticator as stauth

# Load the YAML configuration file
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Initialize the authenticator
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['pre-authorized']
)

# Update the login method to use the 'fields' parameter
name, authentication_status, username = authenticator.login("main", fields={
    "Form name": "Login",
    "Username": "Username",
    "Password": "Password",
    "Login": "Login"
})

# Initialize the LLM client without a hard-coded model
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

# List policies from the directory
def list_policies(directory):
    policies = []
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            policies.append(filename)
    return policies

if authentication_status:
    # Initialize the Streamlit app
    st.title("University Policies Q&A using Local LLM")
    st.caption("Ask questions based on preloaded university policies stored in markdown format.")

    # Add a logout button
    authenticator.logout("Logout", "sidebar")

    # Read and process policies from the specified directory
    policy_directory = "policies"  # Update with the actual path to your directory
    policy_chunks = read_policies_from_directory(policy_directory)
    df = pd.DataFrame({"text": policy_chunks})
    table = lanceDBConnection(df)

    st.write("Policies successfully loaded and stored in the vector database.")

    # Add a navigation sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["Policy Q&A", "Policy Training", "List Policies"])

    # Add model selection to the sidebar
    available_models = ["lmstudio-ai/gemma-2b-it-GGUF/gemma-2b-it-q8_0.gguf", "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf", "model-3"]  # Add more models as needed
    selected_model = st.sidebar.selectbox("Select LLM model", available_models)

    if page == "Policy Q&A":
        # Policy Q&A interface
        question = st.text_input("Ask a question about the university policies:")
        if question:
            result = table.search(question).limit(5).to_list()
            context = [r["text"] for r in result]

            base_prompt = """You are an AI assistant specialized in understanding and explaining university policies. Your task is to read the user question, consult the provided contexts, and generate an accurate, detailed, and clear response based on the policies. Every answer you provide should include specific citations from the provided contexts in the format "Answer [position]", for example: "Students are entitled to a 2-week break per semester [1][2]." If the provided context does not contain the answer, simply state, "The provided context does not contain the answer."

    User question: {}

    Contexts:
    {}
    """
            prompt = base_prompt.format(question, context)

            response = client.chat.completions.create(
                model=selected_model, 
                messages=[{"role": "system", "content": prompt}],
                temperature=0.7
            )

            st.write("Response from LLM:")
            st.markdown(response.choices[0].message.content, unsafe_allow_html=True)

    elif page == "Policy Training":
        st.header("Policy Training Module")

        questions = [
            "What is the university's policy on remote learning?",
            "Describe the university's equal opportunity policy.",
            "What are the guidelines for student conduct?"
        ]

        for idx, question in enumerate(questions):
            st.subheader(f"Question {idx+1}: {question}")
            user_answer = st.text_area(f"Your answer to Question {idx+1}")

            if st.button(f"Submit Answer to Question {idx+1}"):
                training_prompt = """You are an AI assistant specialized in understanding and explaining university policies. Your task is to provide feedback on the user's answer to the following question:

    Question: {}

    User's answer: {}

    Please provide detailed feedback, citing relevant policies.
    """
                prompt = training_prompt.format(question, user_answer)

                response = client.chat.completions.create(
                    model=selected_model, 
                    messages=[{"role": "system", "content": prompt}],
                    temperature=0.7
                )

                st.write("Feedback from LLM:")
                st.markdown(response.choices[0].message.content, unsafe_allow_html=True)

    elif page == "List Policies":
        st.header("List of University Policies")
        policies = list_policies(policy_directory)
        if policies:
            st.write("The following policies are available:")
            for policy in policies:
                st.write(f"- {policy}")
        else:
            st.write("No policies found.")
else:
    st.error("Username/password is incorrect")
