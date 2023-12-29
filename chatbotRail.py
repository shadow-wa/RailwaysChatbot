from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def create_prompt() -> str:
    '''Create a prompt template'''
    _DEFAULT_TEMPLATE: str = """
    You are a Railways Assistant chatbot. You can answer questions about railways data only. If the user asks anything else, say: "I can only help you with railways questions." 
    """
    return _DEFAULT_TEMPLATE

class Chatbot:
    def __init__(self):
        DB_FAISS_PATH = r'.\\vectorStore\\db_faiss_railways'
        # Load our dataset from the CSV file
        loader = CSVLoader(file_path = r'indianRailwaysData2.csv', encoding='utf8', csv_args={'delimiter': ','})
        data = loader.load()

        # Now split the Indian Railways data into chunks so that it cant fit easily into memory
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        # Tokenizing the data as text chunks
        text_chunks = text_splitter.split_documents(data)
        # download the Sentence Transformer Embedding From Hugging Face
        embeddings = HuggingFaceEmbeddings(model_name=r'sentence-transformers/all-MiniLM-L6-v2')
        #vectorDB = FAISS.load_local(".\\vectorStore\\db_faiss_railways", embeddings)
        # Load 

        # Converting the text chunks into embeddings and saving the embeddings into FAISS Knowledge Base
        self.docSearch = FAISS.from_documents(text_chunks, embeddings)

        self.docSearch.save_local(DB_FAISS_PATH)
        # Chatbot model
        self.llm = CTransformers(model=r'C:\Users\HP\Documents\Codes\pythonCodes\chatBot\models\llama-2-7b-chat.ggmlv3.q4_0.bin',
                            model_type = 'llama',
                            max_new_tokens = 512,
                            temperature =0.5,
                            )
    def startChat(self, message)->str:
        # Study on it 
        qa = ConversationalRetrievalChain.from_llm(self.llm, retriever=self.docSearch.as_retriever())

        chat_history = []
        # Add the prompt template to the user's query
        query_with_prompt = create_prompt() + message
        botReply = qa({"question": query_with_prompt, "chat_history": chat_history})
        print(botReply["answer"], type(botReply["answer"]))
        return botReply["answer"]
