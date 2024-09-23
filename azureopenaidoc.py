import os,json,sys
import pickle,joblib
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
# from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import AzureChatOpenAI,AzureOpenAI,AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader,PDFPlumberLoader, UnstructuredPDFLoader, UnstructuredURLLoader,UnstructuredFileLoader
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from flask import Flask, render_template,request,redirect,url_for
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
# from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
# from sentence_transformers import SentenceTransformer
 
# creds------------------------------------------------------------------------------------------------------------------------------
AZURE_OPENAI_KEY="d260fca347a14f40aa9bd0a8df2b3fbb"
AZURE_DEPLOYMENT_NAME="gpt-4"
AZURE_ENDPOINT="https://italent-dev.openai.azure.com/"
AZURE_OPENAI_API_VERSION="2024-04-01-preview"
AZURE_OPENAI_TYPE = "azure"
os.environ["OPENAI_API_VERSION"] = AZURE_OPENAI_API_VERSION         # "2024-04-01-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_ENDPOINT                # "https://italent-dev.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_KEY               # "d260fca347a14f40aa9bd0a8df2b3fbb"
os.environ["OPENAI_API_TYPE"] = AZURE_OPENAI_TYPE                   # "azure"
os.environ["OPENAI_DEPLOYMENT_NAME"] = AZURE_DEPLOYMENT_NAME        # "gpt-4"
# -------------------------------------------------------------------------------------------------------------------------------------
app = Flask(__name__)

@app.route('/',methods=['GET', 'POST'])
def index():
    doc_file1 = "sample.pdf"
    doc_file2 = "Puranas.pdf"
    doc_file3 = "high_school_story.pdf"
    print("pdf is loading.................")

    pdf_reader1 = PDFPlumberLoader(doc_file2)
    # print(pdf_reader1)
    pdf_loader = pdf_reader1.load()
    # print(pdf_loader[:5])
    # for doc in pdf_loader[:5]:
    #     print(doc.page_content[:1000])

    splitter1 = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".",],
            chunk_size=1000,
            chunk_overlap=0,
        )
    docs = splitter1.split_documents(pdf_loader)
    print("splitting data loading...........")
    # print(docs)
    embedding_azure = AzureOpenAIEmbeddings(
        model = "embed-dev",
    )
    
    print("vector store loading...........")
    vector_store = FAISS.from_documents(docs, embedding_azure)
    print(vector_store.index.ntotal)
    file_path = "document_for_puranas.pkl"
    # Save the FAISS index to a pickle file
    # with open(file_path, "wb") as f:
    #     pickle.dump(vector_store, f)
    llm = AzureChatOpenAI(
            azure_deployment=AZURE_DEPLOYMENT_NAME,
            api_version=AZURE_OPENAI_API_VERSION,
            openai_api_key=AZURE_OPENAI_KEY,
            temperature=0.5,
            deployment_name = "gpt-4",
            max_tokens=500,
            # timeout=None,
            # max_retr
    )  
    query = "what is puranas means?"
    # if query:
    #     if os.path.exists(file_path):
    # with open(file_path, "rb") as f:
    #     vectorstore = pickle.load(f)
    print("vector store file loading...........")
        # print(vectorstore.index.ntotal)

    chain = RetrievalQA.from_llm(llm=llm, retriever=vector_store.as_retriever())
    result = chain.invoke(query)
    return result
    sys.exit()
    # ---------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    app.run(port = '5002',debug=True)