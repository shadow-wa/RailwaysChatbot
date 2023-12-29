from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import sys

DB_FAISS_PATH = r'.\\vectorStore\\db_faiss_railways'
# Load our dataset from the CSV file
loader = CSVLoader(file_path = r'indianRailwaysData2.csv', encoding='utf8', csv_args={'delimiter': ','})
data = loader.load()

# Now split the Indian Railways data into chunks so that it cant fit easily into memory
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
# Tokenizing the data as text chunks
text_chunks = text_splitter.split_documents(data)

print(len(text_chunks))

# download the Sentence Transformer Embedding From Hugging Face
embeddings = HuggingFaceEmbeddings(model_name=r'sentence-transformers/all-MiniLM-L6-v2')
vectorDB = FAISS.load_local(".\\vectorStore\\db_faiss_railways", embeddings)
# Load 

# Converting the text chunks into embeddings and saving the embeddings into FAISS Knowledge Base
docSearch = FAISS.from_documents(text_chunks, embeddings)

docSearch.save_local(DB_FAISS_PATH)

# Querying for the results
# query = "What is the value of GDP per capita of Finland Provided in the data?"

# docs = docSearch.similarity_search(query=query, k=3)
# print(docs)


def create_prompt() -> str:
    '''Create a prompt template'''
    _DEFAULT_TEMPLATE: str = """
    You are the Assistant for Railways that works with the given data of railways.
    Your role is to assist users with questions about railways queries only.
    If the user asks anything other than railways queries, refuse to answer and say, 
    "I can assist you only regarding railways QUERIES."
    """
    return _DEFAULT_TEMPLATE

# Chatbot model
llm = CTransformers(model=r'C:\Users\HP\Documents\Codes\pythonCodes\chatBot\models\llama-2-7b-chat.ggmlv3.q4_0.bin',
                    model_type = 'llama',
                    max_new_tokens = 512,
                    temperature =0.5,
                    )
# Study on it 
qa = ConversationalRetrievalChain.from_llm(llm, retriever=vectorDB.as_retriever())

while True:
    chat_history = []
    
    # query = "What is the value of GDP per capita of Finland Provided in the data?"
    query = input("Input the prompt: ")
    if query in ["exit", "quit"]:
        print("Exiting...")
        break
    if query == '':
        continue
    # Add the prompt template to the user's query
    query_with_prompt = create_prompt() + query
    result = qa({"question": query_with_prompt, "chat_history": chat_history})
    print("Response: ", result["answer"])
