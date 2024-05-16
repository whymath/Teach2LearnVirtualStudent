
import chainlit as cl
from dotenv import load_dotenv
import utils
from langchain_openai import ChatOpenAI


load_dotenv()


start_msg = "Hello! I'm Teach2Learn VirtualStudent, a virtual student peer by Jerry Chiang and Yohan Mathew\n\nYou can choose to upload a PDF, or just start chatting\n"
base_instructions = """
Assume you have mastery in the topic and that the user is a someone who is trying to ensure they have a solid understanding by teaching and explaining the material to you.
Your goal is to ensure that the user understands the concept they are explaining by asking questions to help them learn by teaching rather than explaining things directly to them.
Let the user know if they are correct.  If the user is wrong or off track, you should challenge the user by asking them Socratic questions to guide them back.
If they ask for feedback, you should provide constructive feedback on the whole conversation instead of asking another question.
"""
ai_student_instructions = """
Pretend you are a bumbling student with a poor grasp of the topic, are prone to make mistakes, and the user is your teacher.
Your goal is to get the user to teach you about a topic or concept, and you can ask clarifying questions to help them teach better.
You may lay out a scneario for the teacher to help you thru, such as a homework problem, a scenario you need to resolve, or a piece of text you need help deciphering.
Do not explain the material to them except when they ask you to, and when you do as a bumbling student, you may make mistakes and say something unclear or false.
If they ask for feedback, instead of asking another question, you should provide constructive feedback on how well they grasped the content and did in their teaching, including ways they can improve.
When you make a mistake, if the user does not catch or correct you, make sure you let the user know during the feedback at the end of the session.
"""
openai_chat_model = ChatOpenAI(model="gpt-3.5-turbo")
base_chain = utils.create_base_chain(openai_chat_model, base_instructions)


@cl.on_chat_start
async def start_chat():
    print("Chat started")

    # Set the user session settings
    settings = {
        "current_mode": "base_chain"
    }
    cl.user_session.set("settings", settings)

    # Send a welcome message with action buttons
    actions = [
        cl.Action(name="switch_default", value="switch_default_value", label="Switch back to default mode", description="Switch back to default mode"),
        cl.Action(name="switch_ai_student", value="switch_ai_student_value", label="Switch to bumbling student mode", description="Switch to bumbling student mode"),
        cl.Action(name="upload_pdf", value="upload_pdf_value", label="Upload a PDF", description="Upload a PDF")
    ]
    await cl.Message(content=start_msg, actions=actions).send()


@cl.on_message
async def main(message: cl.Message):
    # Print the message content
    user_query = message.content
    settings = cl.user_session.get("settings")

    # Generate the response from the chain
    if settings["current_mode"] == "rag_chain":
        print("\nUsing RAG chain to answer query:", user_query)        
        rag_chain = settings["rag_chain"]
        query_response = rag_chain.invoke({"question" : user_query})
        query_answer = query_response["response"].content
    elif settings["current_mode"] == "ai_student_chain":
        print("\nUsing AI student chain to answer query:", user_query)
        ai_student_chain = settings["ai_student_chain"]
        query_response = ai_student_chain.invoke({"question" : user_query})
        query_answer = query_response.content
    else:
        print("\nUsing base chain to answer query:", user_query)
        query_response = base_chain.invoke({"question" : user_query})
        query_answer = query_response.content
    
    # Create and send the message stream    
    print('query_answer =', query_answer, '\n')
    msg = cl.Message(content=query_answer)
    await msg.send()


@cl.action_callback("upload_pdf")
async def upload_pdf_fn(action: cl.Action):
    print("\nRunning PDF upload and RAG chain creation")
    settings = cl.user_session.get("settings")

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
    # print("\nUploaded file:", file_uploaded, "\n")
    print("file_uploaded.name =", file_uploaded.name, "; file_uploaded.path =", file_uploaded.path)

    # Create the RAG chain and store it in the user session
    if settings["current_mode"] == "ai_student_chain":
        rag_instructions = ai_student_instructions
    else:
        rag_instructions = base_instructions
    rag_chain = utils.create_rag_chain_from_file(openai_chat_model, rag_instructions, file_uploaded, file_uploaded.name)
    settings["rag_chain"] = rag_chain
    settings["current_mode"] = "rag_chain"
    cl.user_session.set("settings", settings)

    msg = cl.Message(content="Okay, I'm ready for you to teach me from the uploaded PDF file.")
    await msg.send()


@cl.action_callback("switch_default")
async def switch_default_fn(action: cl.Action):
    print("\nSwitching back to default base chain")
    settings = cl.user_session.get("settings")

    settings["rag_chain_available"] = False
    cl.user_session.set("settings", settings)

    msg = cl.Message(content="Okay, I'm back to my default mode. What would you like to try teaching me next?")
    await msg.send()


@cl.action_callback("switch_ai_student")
async def switch_ai_student_fn(action: cl.Action):
    print("\nSwitching to AI student mode")
    settings = cl.user_session.get("settings")

    ai_student_chain = utils.create_ai_student_chain(openai_chat_model, ai_student_instructions)
    settings["ai_student_chain"] = ai_student_chain
    settings["current_mode"] = "ai_student_chain"
    cl.user_session.set("settings", settings)

    msg = cl.Message(content="Okay, I will take on the role of an unsure student. What would you like to try teaching me next?")
    await msg.send()
