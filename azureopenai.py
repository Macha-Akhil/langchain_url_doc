import os,json,sys
import pickle,threading,joblib
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
# from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import AzureChatOpenAI,AzureOpenAI,AzureOpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader,PlaywrightURLLoader,SeleniumURLLoader
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
AZURE_OPENAI_API_EMBEDDINGS_DEPLOYMENT_NAME = "embed-dev"

os.environ["OPENAI_API_VERSION"] = AZURE_OPENAI_API_VERSION         # "2024-04-01-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_ENDPOINT                # "https://italent-dev.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_KEY               # "d260fca347a14f40aa9bd0a8df2b3fbb"
os.environ["OPENAI_API_TYPE"] = AZURE_OPENAI_TYPE                   # "azure"
os.environ["OPENAI_DEPLOYMENT_NAME"] = AZURE_DEPLOYMENT_NAME        # "gpt-4"
# -------------------------------------------------------------------------------------------------------------------------------------
app = Flask(__name__)

# To hold the state and URLs
submitted_urls = []
question_form_visible = False
question = ""
answer = ""

@app.route('/',methods=['GET', 'POST'])
def index():
    global submitted_urls, question_form_visible, question, answer 
    if request.method == 'POST':
        if 'submit_urls' in request.form:
            submitted_urls = [request.form['url1'], request.form['url2'], request.form['url3']]
            process_question(submitted_urls)
            # vector_data = process_question(submitted_urls)
            question_form_visible = True
            return redirect(url_for('index'))      
        if 'submit_question' in request.form:
            question = request.form['question']
            answer = process_answer(question)
            # answer = process_answer(vector_data,question)
            question_form_visible = True
            return redirect(url_for('index'))
    return render_template(
        'index.html',
        question_form_visible=question_form_visible,
        urls_submitted=bool(submitted_urls),
        urls=submitted_urls,
        question=question,
        answer=answer
    ) 
def process_question(urls):
    urls = urls
    loader = UnstructuredURLLoader(urls=urls) 
    # loader = PlaywrightURLLoader(urls=urls)
    # return (urls)
    print("loader loading...........") 
    data = loader.load()
    # return data 
    print("splitting data loading...........")
    splitter1 = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".",],
            chunk_size=1000,
            chunk_overlap=200,
        )
    docs = splitter1.split_documents(data)
    # Faiss 
    embedding_azure = AzureOpenAIEmbeddings(
        model="embed-dev",
    )
    # embeddings_huggingface = HuggingFaceInferenceAPIEmbeddings(
    #     api_key="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", model_name="sentence-transformers/all-MiniLM-l6-v2"
    # )
    print("embedding loading...........")
    vectorstore_azureopenai = FAISS.from_documents(docs, embedding_azure)
    print(vectorstore_azureopenai.index.ntotal)
    print("embbeding is over ..............")
    # return vectorstore_azureopenai
    print("file is loading and saving............")
    vectorstore_azureopenai.save_local("vectordata")
    # sys.exit()

    # file_path = "file_faiss_vector1.pkl" # C:\Users\sandh\OneDrive\Documents\python_AI_Langchain\file_faiss_vector1.pkl
    # # obj = CustomObject()
    # with open(file_path, "wb") as f:
    #     print("file is opening and processing ............")
    #     # pickle.dump(vectorstore_azureopenai, f)
    #     joblib.dump(vectorstore_azureopenai, file_path) 
    #     print("file saved............")
def process_answer(question):
    vector_store = None  # Initialize to avoid UnboundLocalError

    embedding_azure = AzureOpenAIEmbeddings(
        model="embed-dev",
    )
    llm = AzureChatOpenAI(
            azure_deployment=AZURE_DEPLOYMENT_NAME,
            api_version=AZURE_OPENAI_API_VERSION,
            openai_api_key=AZURE_OPENAI_KEY,
            temperature=0.2,
            deployment_name = "gpt-4",
            max_tokens=500,
    )
    # file_path = "file_faiss_vector1.pkl"
    if question:
            # if os.path.exists(file_path):  
            # print("file exists............") 
            # with open(file_path, "rb") as f:
                # vector_store = pickle.load(f)
            print("vector store file loading...........")
            vector_store = FAISS.load_local("vectordata",embedding_azure ,allow_dangerous_deserialization=True)
            print("vector store file loading...........")
            chain = RetrievalQA.from_llm(llm=llm, retriever=vector_store.as_retriever())
            result = chain.invoke(question)
            # print(result)
            return result["result"]

if __name__ == "__main__":
    # app.run( debug=True  )
    app.run( port="6001", debug=True )


