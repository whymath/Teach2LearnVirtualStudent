
import chainlit as cl
from dotenv import load_dotenv
import utils
from langchain_openai import ChatOpenAI


load_dotenv()


start_msg = "Hello! I'm Teach2Learn VirtualStudent, a virtual student peer by Jerry Chiang and Yohan Mathew\n\nYou can choose to upload a PDF, or just start chatting\n"
base_instructions = """
Assume you are a virtual student being taught by the user. Your goal is to ensure that the user understands the concept they are explaining.
You should always first let the user know if they are correct or not, and then ask them questions to help them learn by teaching rather than explaining things to them.
If they ask for feedback, you should provide constructive feedback on the whole conversation instead of asking another question.
"""
openai_chat_model = ChatOpenAI(model="gpt-3.5-turbo")
base_chain = utils.create_base_chain(openai_chat_model, base_instructions)


@cl.on_chat_start
async def start_chat():
    print("Chat started")

    # Set the user session settings
    settings = {
        "rag_chain_available": False
    }
    cl.user_session.set("settings", settings)

    # Send a welcome message with action buttons
    actions = [
        cl.Action(name="upload_pdf", value="upload_pdf_value", label="Upload a PDF", description="Upload a PDF"),
        cl.Action(name="switch_default", value="switch_default_value", label="Switch back to default mode", description="Switch back to default mode")
    ]
    await cl.Message(content=start_msg, actions=actions).send()


@cl.on_message
async def main(message: cl.Message):
    # Print the message content
    user_query = message.content
    settings = cl.user_session.get("settings")

    # Generate the response from the chain
    if settings["rag_chain_available"]:
        print("\nUsing RAG chain to answer query", user_query)        
        rag_chain = settings["rag_chain"]
        query_response = rag_chain.invoke({"question" : user_query})
        query_answer = query_response["response"].content
    else:
        print("\nUsing base chain to answer query", user_query)
        query_response = base_chain.invoke({"question" : user_query})
        query_answer = query_response.content
    
    # Create and send the message stream    
    print('query_answer =', query_answer, '\n')
    msg = cl.Message(content=query_answer)
    await msg.send()


@cl.action_callback("upload_pdf")
async def upload_pdf_fn(action: cl.Action):
    print("\nRunning PDF upload and RAG chain creation")

    # Wait for the user to upload a file
    files = None
    while files == None:
        files = await cl.AskFileMessage(
            content="Processing your file...",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()
    file_uploaded = files[0]
    print("\nUploaded file:", file_uploaded, "\n")

    # Create the RAG chain and store it in the user session
    rag_chain = utils.create_rag_chain_from_file(openai_chat_model, base_instructions, file_uploaded.path, file_uploaded.name)
    settings = cl.user_session.get("settings")
    settings["rag_chain"] = rag_chain
    settings["rag_chain_available"] = True
    cl.user_session.set("settings", settings)

    msg = cl.Message(content="Ready to discuss the uploaded PDF file!")
    await msg.send()


@cl.action_callback("switch_default")
async def switch_default_fn(action: cl.Action):
    print("\nSwitching back to default base chain")

    settings = cl.user_session.get("settings")
    settings["rag_chain_available"] = False
    cl.user_session.set("settings", settings)

    msg = cl.Message(content="Okay, I'm back to answering general questions. What would you like to try teaching me next?")
    await msg.send()
