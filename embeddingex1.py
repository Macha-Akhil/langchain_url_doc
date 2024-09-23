import os,json
import pickle,joblib,faiss
import asyncio
import numpy as np
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import AzureChatOpenAI,AzureOpenAI,AzureOpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader,SeleniumURLLoader,PlaywrightURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from flask import Flask, render_template,request
from langchain_community.vectorstores import FAISS
# from langchain_chroma import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
# from langchain.schema import HumanMessage
# from sentence_transformers import SentenceTransformer
# import torchn
# from langchain.schema import Document
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI


app = Flask(__name__)

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
@app.route('/',methods=['GET', 'POST'])
def home():

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
    print("llm loading...........")
    # msg = HumanMessage(content="Explain step by step. How old is the president of USA?")
    # print(llm_chat(messages=[msg]))
    # return str(llm_chat(messages=[msg]))
 
    # result = llm_chat.invoke("Explain step by step. How old is the president of USA?")
    # return str(result)
    # return str(result['content']) # AIMessage obejct is not subscriptable
    # huggingface llm ------------------------------------------------------------------------------------------
    # os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    # repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    # huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    # llm_hug = HuggingFaceEndpoint(
    #     repo_id=repo_id,
    #     temperature=0.1,
    #     huggingfacehub_api_token=huggingfacehub_api_token,
    # )
    text = "translate English to German: How old are you?"
    # return dict(llm(text['content']))

    #-------------------------------------------------------------------------------------
    url1 = "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-february-8-2023"
    url2 = "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-february-9-2023"
    # url3 = "https://indianexpress.com/section/india/"
    url4 = "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-february-10-2023"
    urls = [url1,url2,url4] 

    # palywright , Selenium , unstructured url loader -----------------------------------------
    # loader = UnstructuredURLLoader(urls=urls)  # 
    # loader = SeleniumURLLoader(urls=urls)
    loader = PlaywrightURLLoader(urls=urls)
    print("data loading...........")
    data = loader.load()
    # return dict(data[0]) 
    splitter1 = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".",],
            chunk_size=1000,
            chunk_overlap=200,
        )
    docs = splitter1.split_documents(data)
    print("splitting data loading...........")
    # return str(len(docs))
    # docs is in document format below is in text format ----------------------------------------------------------
    def documents_to_serializable(docs):
    # Convert each Document object to a dictionary
        return [doc.page_content for doc in docs]  
    # Convert documents to a serializable format
    serializable_texts = documents_to_serializable(docs)
    # return str(len(serializable_texts))
    # return serializable_texts
    # ------------------------------------------------------------------------------------------------------------
    # return docs
    # return str(len(docs))
    # return dict(docs[10])

    # huggingface ------------------------------------------------------------------------------------
    # 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 hugging face access 
    # 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
    # text = "The quick brown fox jumps over the lazy dog"
    # embeddings_huggingface = HuggingFaceInferenceAPIEmbeddings(
    #     api_key="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", model_name="sentence-transformers/all-MiniLM-l6-v2"
    # )
    # # result1 = embeddings.embed_query(text)
    # # return result1
    # def document_embedding(docs):
    #     texts = [doc.page_content for doc in docs]
    #     embeddings_result = embeddings_huggingface.embed_documents(texts)
    #     return embeddings_result
    # embedding_result = document_embedding(docs)
    # return embedding_result

    # sentence transformer ---------------------------------------------------------------------------------------------
    # encoder  = SentenceTransformer('all-mpnet-base-v2')
    # def document_embedding(docs):
    #     texts = [doc.page_content for doc in docs]
    #     embeddings_result_sentence = encoder.encode(texts) 
    #     return embeddings_result_sentence
    # embedding_result_sen = document_embedding(docs)
    # return embedding_result_sen

    #AzureOpenAI Embeddings------------------------------------------------------------------------------------------------------------

    # text-embedding-ada-002 # text-embedding-3-large # text-embedding-3-small
    # embeddings = AzureOpenAIEmbeddings(
    #     model="text-embedding-3-large",
    #     # openai_api_version="2024-04-01-preview",
    # )
    # text1 = "The quick brown fox jumps over the lazy dog"
    # code_2 = embeddings.embed_query(text1)
    # return code_2
    # # print((response.data[0].embedding))
    # def document_embedding(docs):
    #     texts = [doc.page_content for doc in docs]
    #     embeddings_res = embeddings.embed_documents(texts)
    #     return embeddings_res
    # Az_embedding_result = document_embedding(docs)
    # return Az_embedding_result
    # ERROR : 
    # openai.NotFoundError: Error code: 404 
    # - {'error': {'code': 'DeploymentNotFound', 
    # 'message': 'The API deployment for this resource does not exist. 
    # If you created the deployment within the last 5 minutes, please wait a moment and try again.'}}
#-----------------------------------------------------------------------------------------------------------------------------------
    # Faiss 
    embedding_azure = AzureOpenAIEmbeddings(
        model = "embed-dev",
        # model="text-embedding-3-large",
        # AZURE_OPENAI_API_EMBEDDINGS_DEPLOYMENT_NAME = "embed-dev",
    )
    # embeddings_huggingface = HuggingFaceInferenceAPIEmbeddings(
    #     api_key="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", model_name="sentence-transformers/all-MiniLM-l6-v2"
    # )
    # vector_store1 = embedding_azure.embed_documents(serializable_texts)
    # print(len(vector_store1[1]))
    # print(len(vector_store1))
    # print(vector_store.dim)
    # return vector_store1[:5]
    print("vector store loading...........")
    vector_store = FAISS.from_documents(docs, embedding_azure)
    # print("Number of vectors:", vector_store.index.ntotal)
    # print("Index dimensions:", vector_store.index.d)
    # print("Index type:", type(vector_store.index))
    # return [vector_store1,vector_store]
    # print(type(vector_store))
    # print(dir(vector_store))
    # return ('hello')
    # Create a FAISS index (example)
    # dimentions_vectors = vector_store.index.d
    # #--------------------------------------------------------------------------------------------------------------
    # # dimension = 128  # Example dimension
    # index = faiss.IndexFlatL2(dimentions_vectors)  # Use your own index initialization
    # # Add some vectors (example)
    # vectors = np.random.random((10, dimentions_vectors)).astype('float32')
    # index.add(vectors)
    #--------------------------------------------------------------------------------------------------------------
    # vector_store = np.array(vector_store)  
    file_path = "file_faiss_vector"
    file_path1 = "file_faiss_vector1.index"
    vector_store.save_local("vectorstore.pkl")
    # faiss.write_index(vector_store.index,file_path1)
    # Save the FAISS index to a pickle file
    # if os.path.exists(file_path):
    # # Delete the file if it exists
    #     print("file exist so del it............")
    #     os.remove(file_path)
    # # Extract index data
    # # index_data = faiss.serialize_index(index)
    # print("file is create and open for writing............")
    # with open(file_path, "wb") as f:        
    #     joblib.dump(vector_store, f)  
    query = "what russia said?"
    # if query:
    #     if os.path.exists(file_path):
    # with open(file_path, "rb") as f:
    #     print("file is exist so open it............")
    #     vectorstore = joblib.load(f)
        # vectorstore = faiss.read_index(file_path)
        # Rebuild the FAISS index
        # loaded_index = faiss.deserialize_index(vectorstore)
    # vectorstore = faiss.read_index(file_path1)
    print("vector store file loading...........")
    vectorstore  = FAISS.load_local("vectorstore.pkl",embeddings=embedding_azure,  allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()
    # retriever = vectorstore.
    chain = RetrievalQA.from_llm(llm=llm, retriever=retriever)
    result = chain.invoke(query)
    return result

        # system_prompt = (
        #     "Use the given context to answer the question. "
        #     "If you don't know the answer, say you don't know. "
        #     "Use three sentence maximum and keep the answer concise. "
        #     "Context: {context}"
        # )
        # prompt = ChatPromptTemplate.from_messages(
        #     [
        #         ("system", system_prompt),
        #         ("human", "{input}"),
        #     ]
        # )
        
        # retriver = vectorstore.as_retriever()
        # question_answer_chain = create_stuff_documents_chain(llm, prompt)
        # chain = create_retrieval_chain(retriever=retriver)
        # result = chain({"input": query})
        # return result

        # # chain.invoke({"input": query})
        # chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        # print("chain loading...........")
            
        # # return_only_outputs=True
        # result_llm = chain.run(query)
        # answer = result_llm["answer"]
        # # # answer = result_llm.get("answer", "no answer found")
        # # return answer
        # return dict(chain.invoke({"question": query}))
                    
        # return(chain({"question": query}, return_only_outputs=True))

        # result_llm = chain.invoke(query)
        # print(result_llm)
        # # return [result_llm.get("answer","no answer found")]
        # # result will be a dictionary of this format --> {"answer": "", "sources": [] }

if __name__ == "__main__":
    app.run( port="5005", debug=True )
