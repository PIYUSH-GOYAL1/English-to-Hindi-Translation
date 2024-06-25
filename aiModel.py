import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS

load_dotenv()


llm = ChatGoogleGenerativeAI(model="gemini-pro" , google_api_key = os.getenv("GOOGLE_API_KEY")
)

# loader = CSVLoader(file_path="updated_data.csv" , source_column = "english")
# data = loader.load()

embeddings = HuggingFaceEmbeddings()

vectordb_file_path = "faiss_index2"

def create_vector_db():
    # Load data from FAQ sheet
    loader = CSVLoader(file_path='updated_data.csv', source_column="english")
    data = loader.load()

    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data,
                                    embedding=embeddings)

    # Save vector database locally
    vectordb.save_local(vectordb_file_path)


file_path = "faiss_index"

def get_chain():
    vector_db = FAISS.load_local(file_path , embeddings = embeddings , allow_dangerous_deserialization = True)
    retriever = vector_db.as_retriever()
    prompt_template = """Given the following sentence in english, generate an new sentence in hindi based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""


    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    chain = RetrievalQA.from_chain_type(llm = llm , chain_type="stuff" , retriever = retriever , input_key = "query" , return_source_documents = True , chain_type_kwargs=chain_type_kwargs)

    return chain

if __name__ == "__main__":
    create_vector_db()
    chain = get_chain()
    print(chain("I want to become a astronaut."))

