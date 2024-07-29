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
import pypdf
import docx

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

# Function to extract text from PDF using pypdf
def extract_text_from_pdf(pdf_file):
    reader = pypdf.PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from DOCX
def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    return text

# Function to generate summary
def generate_summary(text):
    response = client.completions.create(
        model=selected_model,
        prompt=f"Summarize the following text:\n\n{text}",
        max_tokens=150
    )
    summary = response.choices[0].text.strip()
    return summary

# Recursive Text Splitter
def recursive_text_splitter(text, max_chunk_length=1000, overlap=100):
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

def list_policies(directory):
    policies = []
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            policies.append(filename)
    return policies

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

if authentication_status:
    # Initialize the Streamlit app
    st.title("University Policies Q&A and Summarization")
    st.caption("Upload policy documents to get concise summaries and ask questions about university policies.")

    # Add a logout button
    authenticator.logout("Logout", "sidebar")

    # Add a navigation sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["Policy Q&A", "Policy Training", "List Policies"])

    # Add model selection to the sidebar
    available_models = [
        "lmstudio-ai/gemma-2b-it-GGUF/gemma-2b-it-q8_0.gguf",
        "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
        "TheBloke/deepseek-coder-6.7B-instruct-GGUF/deepseek-coder-6.7b-instruct.Q4_K_S.gguf"
    ]
    selected_model = st.sidebar.selectbox("Select LLM model", available_models)

    if page == "Policy Q&A":
        st.header("Policy Q&A")
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
            feedback = st.text_area("Provide feedback on this response:")
            if st.button("Submit Feedback"):
                with open("feedback_log.txt", "a") as feedback_file:
                    feedback_file.write(f"{datetime.now()} - {username} - {feedback}\n")
                st.success("Thank you for your feedback!")
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
        policies = list_policies("policies")  # Update with the actual path to your directory
        if policies:
            st.write("The following policies are available:")
            for policy in policies:
                st.write(f"- {policy}")
        else:
            st.write("No policies found.")

    st.header("Upload Policy Document")
    uploaded_file = st.file_uploader("Upload a policy document", type=["pdf", "docx", "txt"])
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = extract_text_from_docx(uploaded_file)
        else:
            text = uploaded_file.read().decode("utf-8")

        st.subheader("Document Text")
        st.text_area("Document Text", text, height=300)

        if st.button("Generate Summary"):
            summary = generate_summary(text)
            st.subheader("Summary")
            st.write(summary)

    # Load and process policies from the specified directory
    policy_chunks = read_policies_from_directory("policies")
    df = pd.DataFrame({"text": policy_chunks})
    table = lanceDBConnection(df)

else:
    st.error("Username/password is incorrect")
