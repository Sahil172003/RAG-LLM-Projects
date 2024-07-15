import os
import streamlit as st
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

anthropic_api_key = "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"

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
    
def main():
    st.header("Chat with PDF 💬")

    # Upload PDF
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Text splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Embeddings
        store_name = pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = HuggingFaceEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        # Anthropic LLM
        llm = ChatAnthropic(model="claude-3-sonnet-20240229", anthropic_api_key=anthropic_api_key)

        # User query
        query = st.text_input("Ask questions about your PDF file:")
        submit = st.button("Submit")
        if query and submit:
            retriever = VectorStore.as_retriever()
            
            # Create a prompt template
            prompt = ChatPromptTemplate.from_template("""Answer the following question based on the given context:

            Context: {context}

            Question: {question}

            Answer:""")

            # Set up the RAG chain
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            response = rag_chain.invoke(query)
            st.write(response)
if __name__ == '__main__':
    main()