import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import  GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.docstore.document import Document

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="Chat with PDF - Gemini", layout="wide")

st.markdown(
    """
    <style>
        /* Global Style */
        body {
            background-color: #0e1117;
            color: #ffffff;
        }

        /* Sidebar Customization */
        [data-testid="stSidebar"] {
            background-color: #161b22;
        }

        /* Custom Header */
        .main-title {
            font-size: 32px;
            font-weight: bold;
            text-align: center;
            color: #58a6ff;
        }

        /* Response Box */
        .response-box {
            background-color: #161b22;
            padding: 15px;
            border-radius: 10px;
            margin-top: 10px;
            box-shadow: 0px 4px 10px rgba(255, 255, 255, 0.1);
        }

        /* Chat Input */
        input {
            background-color: #0d1117;
            color: white;
            border-radius: 5px;
            padding: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

def get_pdf_text(pdf_docs):
    all_text = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                all_text.append((text, page_num + 1))  # Store page number (1-based index)
    return all_text

def get_text_chunks(text_with_pages):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    
    chunks = []
    metadata_list = []  # Store metadata separately

    for text, page_num in text_with_pages:
        split_texts = text_splitter.split_text(text)
        for chunk in split_texts:
            chunks.append(chunk)  # Only store text for FAISS
            metadata_list.append({"page": page_num})  # Store page numbers

    return chunks, metadata_list

def get_vector_store(text_chunks, metadata_list):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Use `from_texts` with metadata
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadata_list)

    return vector_store

def get_conversational_chain():
    prompt_template = """
                        You are an expert AI research analyst with deep domain knowledge. Your task is to provide comprehensive, insightful answers by analyzing the provided context carefully.

                        ### Instructions:
                        1. ANALYZE the context thoroughly before responding
                        2. STRUCTURE your response in a clear, logical format
                        3. SYNTHESIZE information from multiple parts of the context when relevant
                        4. HIGHLIGHT key concepts using markdown formatting
                        5. CITE specific parts of the context to support your points
                        6. ACKNOWLEDGE knowledge gaps explicitly

                        ### Response Guidelines:
                        - Start with a high-level summary of your findings
                        - Break down complex topics into digestible sections
                        - Use bullet points and numbered lists for clarity when appropriate
                        - Include relevant examples or data points from the context
                        - Explain technical terms if they're crucial to understanding
                        - Be explicit about any limitations in the available information

                        ### Context:
                        {context}

                        ### User Question:
                        {question}

                        ### Expert Analysis:
                        I've analyzed the provided context and will now provide a detailed response:

                        """

    
    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   temperature=0.3)
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context","question"])

    chain = load_qa_chain(model,chain_type="stuff",prompt=prompt)

    return chain

def user_input(user_question, vector_store, top_k=2):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Retrieve top-k most relevant document chunks
    docs = vector_store.similarity_search(user_question, k=top_k)
    
    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    # Extract only the **most relevant** page numbers
    matched_pages = sorted(set(doc.metadata["page"] for doc in docs if "page" in doc.metadata))

    return response["output_text"], matched_pages

st.markdown("<h1 class='main-title'>💬 Chat with Your PDF Using Gemini</h1>", unsafe_allow_html=True)

def main():
    # st.set_page_config(page_title="Chat PDF")
    # st.header("Chat with PDF using Gemini")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text_with_pages = get_pdf_text(pdf_docs)
                text_chunks, metadata_list = get_text_chunks(raw_text_with_pages)
                vector_store = get_vector_store(text_chunks, metadata_list)
                
                # Save to session state
                st.session_state["vector_store"] = vector_store
                
                st.success("Processing Complete!")

    # User input
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        if "vector_store" in st.session_state:
            response, pages = user_input(user_question, st.session_state["vector_store"])
            
            # Display result
            st.write("### Reply:")
            st.write(response)
            
            if pages:
                st.markdown(
                    f'<p style="color:#FFD700; font-size:20px;">📄 <b>Answer found on page(s): {", ".join(map(str, pages))}</b></p>',
                    unsafe_allow_html=True
                )
            else:
                st.write("⚠️ No exact page reference found.")
        else:
            st.warning("Please upload and process a PDF first.")

if __name__ == "__main__":
    main()