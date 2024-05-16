
import chainlit as cl
from dotenv import load_dotenv
import utils


load_dotenv()


start_msg = "Teach2Learn Virtual Student by Jerry Chiang and Yohan Mathew\n\nYou can choose to upload a PDF, or just start chatting"

# Create the RAQA chain and store it in the user session
raqa_chain = utils.create_raqa_chain_from_docs()


@cl.on_chat_start
async def start_chat():
    # # Create the RAQA chain and store it in the user session
    # raqa_chain = utils.create_raqa_chain_from_docs()
    # settings = {
    #     "chain": raqa_chain
    # }
    # cl.user_session.set("settings", settings)
    print("Chat started")

    # Send a welcome message with an action button
    actions = [
        cl.Action(name="upload_pdf", value="upload_pdf_value", label="Upload a PDF", description="Upload a PDF")
    ]
    await cl.Message(content=start_msg, actions=actions).send()


@cl.on_message
async def main(message: cl.Message):
    # Print the message content
    user_query = message.content
    print('\nuser_query =', user_query)

    # Get the chain from the user session
    try:
        settings = cl.user_session.get("settings")
        raqa_chain_upload = settings["raqa_chain_upload"]
    except Exception as e:
        print("Error fetching chain from session, defaulting to base chain", e)
        raqa_chain_upload = None

    # Generate the response from the chain
    if raqa_chain_upload:
        print("\nUsing UPLOAD chain to answer query", user_query)
        query_response = raqa_chain_upload.invoke({"question" : user_query})
    else:
        print("\nUsing DEFAULT chain to answer query", user_query)
        query_response = raqa_chain.invoke({"question" : user_query})
    query_answer = query_response["response"].content
    print('query_answer =', query_answer, '\n')
    
    # Create and send the message stream
    msg = cl.Message(content=query_answer)
    await msg.send()


@cl.action_callback("upload_pdf")
async def upload_pdf_fn(action: cl.Action):
    print("\nThe user clicked on an action button!")

    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Processing your file",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file_uploaded = files[0]
    print("\nUploaded file:", file_uploaded, "\n")

    # Create the RAQA chain and store it in the user session
    filepath_uploaded = file_uploaded.path
    filename_uploaded = file_uploaded.name
    raqa_chain_upload = utils.create_raqa_chain_from_file(filepath_uploaded, filename_uploaded)

    settings = {
        "raqa_chain_upload": raqa_chain_upload
    }
    cl.user_session.set("settings", settings)

    msg = cl.Message(content="Thank you for uploading!")
    await msg.send()
