
import chainlit as cl
from dotenv import load_dotenv
import utils


load_dotenv()


@cl.on_chat_start
async def start_chat():
    # Create the RAQA chain and store it in the user session
    raqa_chain = utils.create_raqa_chain_from_docs()
    settings = {
        "chain": raqa_chain
    }
    cl.user_session.set("settings", settings)


@cl.on_message
async def main(message: cl.Message):
    # Print the message content
    user_query = message.content
    print('user_query =', user_query)

    # Get the chain from the user session
    settings = cl.user_session.get("settings")
    raqa_chain = settings["chain"]

    # Generate the response from the chain
    query_response = raqa_chain.invoke({"question" : user_query})
    query_answer = query_response["response"].content
    print('query_answer =', query_answer)
    
    # Create and send the message stream
    msg = cl.Message(content=query_answer)
    await msg.send()
