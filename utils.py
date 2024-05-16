import tiktoken
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
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


def create_base_chain(openai_chat_model, base_instructions):
    human_template = "{question}"
    base_prompt = ChatPromptTemplate.from_messages([
        ("system", base_instructions),
        ("human", human_template)
    ])
    base_chain = base_prompt | openai_chat_model
    print("Created base chain\n")
    return base_chain


def create_rag_chain_from_file(openai_chat_model, base_instructions, file_path, file_name):

    # Load the documents from a PDF file using PyMuPDFLoader
    docs = PyMuPDFLoader(file_path).load()
    print("Loaded", len(docs), "documents")
    print("First document:\n", docs[0], "\n")

    # Create a Qdrant vector store from the split chunks and embedding model, and obtain its retriever
    split_chunks = chunk_documents(docs, tiktoken_len)
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    qdrant_vectorstore = Qdrant.from_documents(
        split_chunks,
        embedding_model,
        location=":memory:",
        collection_name=file_name,
    )
    qdrant_retriever = qdrant_vectorstore.as_retriever()
    print("Created Qdrant vector store from uploaded PDF file =", file_name)

    # Define the RAG prompt template
    RAG_PROMPT = """
    Use the provided context while replying to the user query. Only use the provided context to answer the query.

    QUERY:
    {question}

    CONTEXT:
    {context}
    """
    RAG_PROMPT = base_instructions + RAG_PROMPT
    print("RAG prompt template =", RAG_PROMPT)
    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

    # Create the retrieval augmented QA chain using the Qdrant retriever, RAG prompt, and OpenAI chat model
    rag_chain = (
        {"context": itemgetter("question") | qdrant_retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": rag_prompt | openai_chat_model, "context": itemgetter("context")}
    )
    print("Created RAG chain from uploaded PDF file =", file_name, "\n")

    return rag_chain
