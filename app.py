
import chainlit as cl
from dotenv import load_dotenv
import utils

from openai import AsyncOpenAI
import time


load_dotenv()


start_msg = "Teach2Learn Virtual Student by Jerry Chiang and Yohan Mathew\n\nYou can choose to upload a PDF, or just start chatting"
# instructions = "You are a helpful assistant"
instructions = "You are a virtual student being taught by the user. You can ask clarifying questions to better understand the user's explanation. Your goal is to ensure that the user understands the concept they are explaining. You can also ask questions to help the user elaborate on their explanation. You can ask questions like 'Can you explain that in simpler terms?' or 'Can you provide an example?'."
client = AsyncOpenAI()
assistant = client.beta.assistants.create(
    name="T2L Virtual Student",
    instructions=instructions,
    model="gpt-3.5-turbo",
)
print("assistant =", assistant)
thread = client.beta.threads.create()
print("thread =", thread)


@cl.on_chat_start
async def start_chat():

    # Create an OpenAI assistant
    # instructions = "You are a helpful assistant"
    # client = AsyncOpenAI()
    # assistant = client.beta.assistants.create(
    #     name="T2L Virtual Student",
    #     instructions=instructions,
    #     model="gpt-3.5-turbo",
    # )
    # thread = client.beta.threads.create()

    # Store the assistant and thread in the user session
    # settings = {
    #     "instructions": instructions,
    #     "client": client,
    #     "assistant": assistant,
    #     "thread": thread
    # }
    settings = {}
    cl.user_session.set("settings", settings)

    # Send a welcome message with an action button
    actions = [
        cl.Action(name="upload_pdf", value="upload_pdf_value", label="Upload a PDF")
    ]
    await cl.Message(content=start_msg, actions=actions).send()


@cl.on_message
async def main(message: cl.Message):
    # Print the message content
    user_query = message.content
    print('user_query =', user_query)

    # Get the chain from the user session
    settings = cl.user_session.get("settings")
    # instructions = settings["instructions"]
    # client = settings["client"]
    # assistant = settings["assistant"]
    # thread = settings["thread"]
    try:
        raqa_chain = settings["raqa_chain"]
    except KeyError:
        raqa_chain = None

    # Generate the response from the chain
    if raqa_chain:
        print("Using RAQA chain to generate response")
        query_response = raqa_chain.invoke({"question" : user_query})
        query_answer = query_response["response"].content
        print('query_answer =', query_answer)
    else:
        print("Using OpenAI assistant to generate response")
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_query
        )
        print("message =", message)
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
            instructions=instructions
        )
        print("run =", run)
        while run.status == "in_progress" or run.status == "queued":
            time.sleep(1)
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
        print("run.status =", run.status)
        messages = client.beta.threads.messages.list(
            thread_id=thread.id
        )
        # print("messages =", messages)
        print("messages.data =", messages.data)
        query_answer = messages.data[0].content
    
    # Create and send the message stream
    msg = cl.Message(content=query_answer)
    await msg.send()


@cl.action_callback("upload_pdf")
async def upload_pdf_fn(action: cl.Action):
    print("The user clicked on the action button!")

    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Waiting for file selection",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(
        content=f"Processing `{file.name}`...", disable_human_feedback=True
    )
    await msg.send()

    # Create the RAQA chain and store it in the user session
    raqa_chain = utils.create_raqa_chain_from_docs(file)
    settings = {
        "raqa_chain": raqa_chain
    }
    cl.user_session.set("settings", settings)

    return "Thank you for clicking on the action button!"
