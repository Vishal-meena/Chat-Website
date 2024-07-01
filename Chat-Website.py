import streamlit as st 
from langchain.chat_models import ChatOpenAI 
import openai 
from langchain import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
load_dotenv()



def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    documents = loader.load()

    # we split the documents into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents_chunks = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    my_activeloop_org_id = "vishalug3016"
    my_activeloop_dataset_name = "Brainlox_course_data"
    dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
    vector_store = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
    vector_store.add_documents(documents_chunks)

    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user","{input}"),
        ("user"," You are an exceptional customer support chatbot that gently answer questions")
    ])

    retriever_chain = create_history_aware_retriever(llm,retriever,prompt)
    return retriever_chain

def get_conversational_chain(retriever_chain):
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages(
        [("system","Answer the user's questions based on the below context: \n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
         ("user", "{input}"), 
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)

    return create_retrieval_chain(retriever_chain,stuff_documents_chain)
    
def get_response(user_query):
    # create conversation chain
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input":user_query
    })
    return response['answer']

# App Config 
st.set_page_config(page_title = "Chat with website")
st.title("Chat With Website")


# Sidebar 
with st.sidebar:  
    st.header("Settings") 
    website_url = st.text_input("website_url")

if website_url is None or website_url=="":
    st.info("Please Enter a website URL")

else:

    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [SystemMessage(content = "Hello, I am a bot. How can i help you?"),] 
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)


    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query !="":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content = user_query))
        st.session_state.chat_history.append(SystemMessage(content = response))


    # Conversation 
    for message in st.session_state.chat_history:
        if isinstance(message,SystemMessage):
            with st.chat_message("System"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
