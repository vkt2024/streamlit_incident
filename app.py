# File: interactive_rag_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import os
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv
from pathlib import Path
import mwparserfromhell
from io import StringIO  # For JSON parsing
import logging
import re


# Configure logging to a file
logging.basicConfig(level=logging.ERROR, filename="error.log", filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]


# Initialize OpenAI model
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    openai_api_key=OPENAI_API_KEY
)

# Streamlit App Configuration
st.title("📄 Intelligent Document Analysis Tool")
st.sidebar.title("⚙️ Options")
st.sidebar.info("Upload your wikitext document database to begin.")


# Function to load and parse wikitext files
@st.cache_data
def load_wikitext_documents(folder_path):
    documents = []
    for file_path in Path(folder_path).glob("*.wikitext"):
        with open(file_path, "r", encoding="utf-8") as f:
            raw_content = f.read()
            parsed = mwparserfromhell.parse(raw_content)
            plain_text = parsed.strip_code()
            documents.append(Document(page_content=plain_text, metadata={"source": file_path.name}))
    return documents


# Function to extract incidents using OpenAI
def extract_incidents_with_openai(documents):
    extracted_incidents = []
    for doc in documents:
        # Extract year from the file name
        file_name = doc.metadata["source"]
        year_match = re.search(r"\b(19|20)\d{2}\b", file_name)
        file_year = year_match.group(0) if year_match else None

        prompt = f"""
        Analyze the following document and extract each and all incident details in the format of a structured JSON array.
        Each entry in the JSON should include:
        - "Date": The date of the incident (if mentioned).
        - "Month": The month of the incident.
        - "Year": The year of the incident. If the year is missing or unclear, infer it as {file_year}.
        - "Type": A very short description of the type of incident. 

        Document Content:
        {doc.page_content}

        Please provide only the JSON array as the response.
        """
        response = llm.invoke(prompt)
        response_text = response.content  # Extract the text content from AIMessage

        try:
            # Parse the JSON response
            incidents = pd.read_json(StringIO(response_text))

            # Add the file's extracted year if the Year field is missing or invalid
            if "Year" in incidents.columns:
                incidents["Year"] = incidents["Year"].fillna(file_year).astype(str)
                incidents["Year"] = incidents["Year"].apply(
                    lambda x: file_year if not re.match(r"^\d{4}$", str(x)) else x)
            else:
                incidents["Year"] = file_year  # Use file year if Year is completely missing

            # Add the source file name as metadata
            incidents["Source"] = file_name
            extracted_incidents.append(incidents)
        except ValueError as e:
            logging.error(f"Failed to extract incidents from {file_name}: {e}")
            logging.error(f"Response Text: {response_text}")
            continue

    # Combine extracted data into a single dataframe
    if extracted_incidents:
        combined_df = pd.concat(extracted_incidents, ignore_index=True)

        # Ensure Year is numeric and remove invalid entries
        combined_df["Year"] = pd.to_numeric(combined_df["Year"], errors="coerce")
        combined_df = combined_df.dropna(subset=["Year"])
        combined_df["Year"] = combined_df["Year"].astype(int)

        return combined_df

    # Return an empty dataframe if no valid incidents were extracted
    return pd.DataFrame()

# Function to set up the RAG system
@st.cache_resource
def setup_rag_system(_documents):
    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(_documents)

    # Create embeddings and vectorstore
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Configure the RAG system
    retriever = vectorstore.as_retriever()
    qa_system = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_system


# Sidebar: Upload document files
uploaded_files = st.sidebar.file_uploader(
    "Upload your wikitext document files:", 
    type=["wikitext"], 
    accept_multiple_files=True
)

if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        # Read the uploaded file content
        raw_content = uploaded_file.read().decode("utf-8")
        parsed = mwparserfromhell.parse(raw_content)
        plain_text = parsed.strip_code()
        documents.append(Document(page_content=plain_text, metadata={"source": uploaded_file.name}))
    
    st.sidebar.success(f"Loaded {len(documents)} documents!")

    # Set up the RAG system
    qa_system = setup_rag_system(documents)

    # Process and extract incidents
    st.header("🔍 Incident Extraction")
    with st.spinner("Extracting incidents from documents..."):
        incident_df = extract_incidents_with_openai(documents)

        if not incident_df.empty:
            st.subheader("Extracted Incidents")
            st.dataframe(incident_df)

            # Add Year and Month columns for analysis
            incident_df["Date"] = pd.to_datetime(incident_df["Date"], errors="coerce")

            # Incident Analysis Dashboard
            st.header("📊 Incident Analysis Dashboards")

            # Drop rows with missing or invalid Year and Month values
            incident_df = incident_df.dropna(subset=["Year", "Month"])

            # Ensure Year contains only valid integers
            incident_df = incident_df[incident_df["Year"].apply(lambda x: str(x).replace('.', '', 1).isdigit())]

            # Ensure Year is an integer
            incident_df["Year"] = incident_df["Year"].astype(float).astype(int)

            # Standardize the Month column
            incident_df["Month"] = incident_df["Month"].str.strip()
            incident_df["Month"] = pd.Categorical(
                incident_df["Month"],
                categories=[
                    "January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"
                ],
                ordered=True
            )

            # Yearly Incident Count
            yearly_counts = incident_df.groupby("Year").size().reset_index(name="Incident Count")
            print(f"Data type of yearly_count is {yearly_counts}.")
            st.subheader("Yearly Incident Count")
            year_chart = px.bar(
                yearly_counts,
                x="Year",
                y="Incident Count",
                title="Yearly Incident Count",
                labels={"Year": "Year", "Incident Count": "Number of Incidents"},
            )

            # Update x-axis tick format to display integers only
            year_chart.update_xaxes(tickformat=".0f")

            st.plotly_chart(year_chart, use_container_width=True)

            # Monthly Incident Count
            # Drop rows with missing Year or Month values
            # incident_df = incident_df.dropna(subset=["Year", "Month"])
            #
            # # Remove invalid values in the Year column (e.g., non-numeric values)
            # incident_df = incident_df[incident_df["Year"].apply(lambda x: str(x).isdigit())]
            #
            # # Ensure Year is an integer
            # incident_df["Year"] = incident_df["Year"].astype(int)
            #
            # # Standardize the Month column
            # incident_df["Month"] = incident_df["Month"].str.strip()
            # incident_df["Month"] = pd.Categorical(
            #     incident_df["Month"],
            #     categories=[
            #         "January", "February", "March", "April", "May", "June",
            #         "July", "August", "September", "October", "November", "December"
            #     ],
            #     ordered=True
            # )

            monthly_counts = incident_df.groupby(["Year", "Month"]).size().reset_index(name="Incident Count")
            monthly_counts["Year"] = monthly_counts["Year"].astype(int)
            st.subheader("Monthly Incident Count")


            month_chart = px.bar(
                monthly_counts,
                x="Month",
                y="Incident Count",
                color="Year",
                title="Monthly Incident Count by Year",
                labels={"Month": "Month", "Incident Count": "Number of Incidents"}, )
            #     category_orders={"Month": [
            #         "January", "February", "March", "April", "May", "June",
            #         "July", "August", "September", "October", "November", "December"
            #     ]},
            # )

            # Ensure Year appears as an integer in the legend
            month_chart.update_layout(coloraxis_colorbar=dict(tickformat=".0f"))

            st.plotly_chart(month_chart, use_container_width=True)

            # Display Trends Over Time
            # Aggregate incidents by date
            daily_counts = incident_df.groupby("Date").size().reset_index(name="Incident Count")

            # Create the line chart
            trend_chart = px.line(
                daily_counts,
                x="Date",
                y="Incident Count",
                title="Incident Trends Over Time",
                labels={"Date": "Date", "Incident Count": "Number of Incidents"},
            )
            st.plotly_chart(trend_chart, use_container_width=True)

            # Set up the RAG system
            st.header("🔍 Retrieval-Augmented Generation (RAG) System")
            qa_system = setup_rag_system(documents)

            query = st.text_input("Ask a question about the documents:")
            if query:
                with st.spinner("Fetching answer..."):
                    response = qa_system({"query": query})
                    st.write("**Answer:**", response["result"])

                    # Display the retrieved source documents
                    st.write("**Sources:**")
                    for doc in response["source_documents"]:
                        st.markdown(f"- **{doc.metadata['source']}**")

            # Display Raw Data Viewer
            # st.header("📂 Data Viewer")
            # st.dataframe(incident_df)
        else:
            st.info("No incidents could be extracted from the documents.")
else:
    st.info("Please enter the path to your document database to get started.")
