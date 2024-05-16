import tiktoken
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain.schema.runnable import RunnablePassthrough
from chainlit.types import AskFileResponse
from langchain.document_loaders import PyPDFLoader


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


def process_file(file: AskFileResponse):
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tempfile:
        with open(tempfile.name, "wb") as f:
            f.write(file.content)

    pypdf_loader = PyPDFLoader(tempfile.name)
    texts = pypdf_loader.load_and_split()
    texts = [text.page_content for text in texts]
    return texts


def create_base_chain(openai_chat_model, system_prompt):
    human_template = "{question}"
    base_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        # Example 1
        # ("human", "I want to teach you about the Pythagorean Theorem. Can you pretend to know the topic well and give me feedback on how well I explain it?"),
        # ("ai", "That sounds great! I’m here to learn about the Pythagorean Theorem from you. Can you explain what the Pythagorean Theorem is and how to apply it?"),
        # Example 2
        # ("human", "The Pythagorean Theorem is a theorem that relates the lengths of right triangles. More specifically, if a triangle has 3 sides - a, b and c, with c being the hypotenuse - then the theorem tells us a^2+b^2 = c^2. This helps us calculate distances in 2-D space and has applications in math, science, engineering, and architecture."),
        # ("ai", "Great! That makes sense. Can you walk me through an example of how to apply the Pythagorean Theorem to a real world problem?"),
        # Example 3
        # ("human", "The Pythagorean Theorem has something to do with triangles and the lengths of their sides. I'm not sure what though."),
        # ("ai", "Okay, I see. What kind of triangles does it deal with? And what is the relationship between the three sides? Maybe this site can help us: https://byjus.com/maths/pythagoras-theorem"),
        # Example 4
        ("human", "I'd like to end the session"),
        ("ai", "No worries.  Would you like me to share some feedback with you?"),
        # Example 5
        # ("human", "I don't want to discuss the Pythagorean Theorem anymore.  Instead, I want to talk more about circles."),
        # ("ai", "That's fine.  Would you like for me to first give you some feedback on this lesson before we switch to another topic?"),
        # Example 6
        ("human", "Can you tell me how I did?"),
        ("ai", "Sure! Shall I first give you some feedback on how well you covered the content, and then some feedback on your approach to teaching?"),
        # Example 7
        ("human", "Can you tell me the answer?"),
        ("ai", "Hmm, maybe we can figure it out together? If I passed you some references to look up, can you help me figure it out?"),
        # Example 8a (mistake)
        ("human", "So using the Pythagorean Theorem, given the hypotenuse is 13 and one of the legs is 5, we know the length of the other leg is going to equal sqrt(13^2 - 6^2) = sqrt(169 - 36) = sqrt(133) which is almost 12?"),
        ("ai", "Hmm, can you explain to me why you have written 6^2 rather than 5^2?"),
        ("human", human_template)
    ])
    base_chain = base_prompt | openai_chat_model
    print("Created base chain\n")
    return base_chain


def create_ai_student_chain(openai_chat_model, system_prompt):
    human_template = "{question}"
    base_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        # Example 3
        # ("human", "The Pythagorean Theorem has something to do with triangles and the lengths of their sides. I'm not sure what though."),
        # ("ai", "Okay, I see. What kind of triangles does it deal with? And what is the relationship between the three sides? Maybe this site can help us: https://byjus.com/maths/pythagoras-theorem"),
        # Example 4
        ("human", "I'd like to end the session"),
        ("ai", "No worries.  Would you like me to share some feedback with you?"),
        # Example 5
        # ("human", "I don't want to discuss the Pythagorean Theorem anymore.  Instead, I want to talk more about circles."),
        # ("ai", "That's fine.  Would you like for me to first give you some feedback on this lesson before we switch to another topic?"),
        # Example 6
        ("human", "Can you tell me how I did?"),
        ("ai", "Sure! Shall I first give you some feedback on how well you covered the content, and then some feedback on your approach to teaching?"),
        # Example 7
        ("human", "Can you tell me the answer?"),
        ("ai", "Hmm, maybe we can figure it out together? If I passed you some references to look up, can you help me figure it out?"),
        # Example 8b (mistake)
        ("human", "So can you show me how you would apply the Pythagorean Theorem to solve this next problem? Let's say you are building a 8 feet tall vertical structure and you'd like to add support beams all around it 6 feet away from its base. Can you help me calculate how long these support beams should be?"),
        ("ai", "Because the structure is vertical and the support beams are on the ground, we see this forms a right triangle. So we can use the Pythagorean Theorem to calculate the length of the support beam. Let's call the length of the support beam 'c', while the height of the vertical structure is 'a' and the distance the support beam is away from the structure is 'b'.  Hence, if c^2 = a^2 + b^2, I think we need to solve for c = sqrt(8^2 + 6^2) = sqrt(16+12) = sqrt (28) = 5.3?  Did I do that right?"),
        ("human", human_template)
    ])
    ai_student_chain = base_prompt | openai_chat_model
    print("Created base chain\n")
    return ai_student_chain


def create_rag_chain_from_file(openai_chat_model, base_instructions, file_response, file_name):

    # Load the documents from a PDF file using PyMuPDFLoader
    docs = PyMuPDFLoader(file_response.path).load()
    # docs = process_file(file_response)
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
    Use the provided context while replying to the user query. Only use the provided context to respond to the query.
    If the context is not sufficient, you can respond with "I cannot seem to find this topic in the PDF. Would you like to switch to back to the default or bumbling student mode?".

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
