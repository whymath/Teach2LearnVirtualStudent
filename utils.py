import tiktoken
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from operator import itemgetter
# from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough


def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-3.5-turbo").encode(
        text,
    )
    return len(tokens)


def chunk_documents(docs, tiktoken_len):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 0,
        length_function = tiktoken_len,
    )

    split_chunks = text_splitter.split_documents(docs)

    print('len(split_chunks) =', len(split_chunks))

    return split_chunks


def create_raqa_chain_from_docs():
    # Load the documents from a PDF file using PyMuPDFLoader
    # docs = PyMuPDFLoader("data/c7318154-f6ae-4866-89fa-f0c589f2ee3d.pdf").load()
    docs = PyMuPDFLoader("https://d18rn0p25nwr6d.cloudfront.net/CIK-0001326801/c7318154-f6ae-4866-89fa-f0c589f2ee3d.pdf").load()

    # Print the number of loaded documents
    print("Loaded", len(docs), "documents")

    # Print the first document
    print(docs[0])

    # Split the documents into chunks based on their length
    split_chunks = chunk_documents(docs, tiktoken_len)

    # Create an instance of the OpenAIEmbeddings model for text embeddings
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    # Create a Qdrant vector store from the split chunks
    qdrant_vectorstore = Qdrant.from_documents(
        split_chunks,
        embedding_model,
        location=":memory:",
        collection_name="Meta 10-k Filings",
    )

    # Create a retriever from the Qdrant vector store
    qdrant_retriever = qdrant_vectorstore.as_retriever()

    # Define the RAG prompt template
    RAG_PROMPT = """
    CONTEXT:
    {context}

    QUERY:
    {question}

    Use the provided context to answer the provided user query. Only use the provided context to answer the query. If you do not know the answer, respond with "I don't know".
    """

    # Create a ChatPromptTemplate instance from the RAG prompt template
    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

    # Create an instance of the ChatOpenAI model
    openai_chat_model = ChatOpenAI(model="gpt-3.5-turbo")

    # Define the retrieval augmented QA chain
    retrieval_augmented_qa_chain = (
        {"context": itemgetter("question") | qdrant_retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": rag_prompt | openai_chat_model, "context": itemgetter("context")}
    )
    print("Created retrieval augmented QA chain from default PDF file")

    return retrieval_augmented_qa_chain


def create_raqa_chain_from_file(filepath_uploaded, filename_uploaded):

    # # Load the documents from a PDF file using PyMuPDFLoader
    # docs = PyMuPDFLoader("https://d18rn0p25nwr6d.cloudfront.net/CIK-0001326801/c7318154-f6ae-4866-89fa-f0c589f2ee3d.pdf").load()
    docs = PyMuPDFLoader(filepath_uploaded).load()
    print("Loaded", len(docs), "documents")
    print(docs[0])

    # Create a Qdrant vector store from the split chunks and embedding model, and obtain its retriever
    split_chunks = chunk_documents(docs, tiktoken_len)
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    qdrant_vectorstore = Qdrant.from_documents(
        split_chunks,
        embedding_model,
        location=":memory:",
        collection_name="LoadedPDF",
    )
    qdrant_retriever = qdrant_vectorstore.as_retriever()

    # Define the RAG prompt template
    # RAG_PROMPT = """
    # Assume you are a virtual student being taught by the user. You can ask clarifying questions to better understand the user's explanation. Your goal is to ensure that the user understands the concept they are explaining. You can also ask questions to help the user elaborate on their explanation. You can ask questions like "Can you explain that in simpler terms?" or "Can you provide an example?".

    # USER MESSAGE:
    # {question}
    # """
    RAG_PROMPT = """
    CONTEXT:
    {context}

    QUERY:
    {question}

    Use the provided context to answer the provided user query. Only use the provided context to answer the query. If you do not know the answer, respond with "I don't know".
    """
    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

    # Create the retrieval augmented QA chain using the Qdrant retriever, RAG prompt, and OpenAI chat model
    openai_chat_model = ChatOpenAI(model="gpt-3.5-turbo")
    retrieval_augmented_qa_chain = (
        {"context": itemgetter("question") | qdrant_retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": rag_prompt | openai_chat_model, "context": itemgetter("context")}
    )
    print("Created retrieval augmented QA chain from uploaded PDF file =", filename_uploaded, "\n")

    return retrieval_augmented_qa_chain
