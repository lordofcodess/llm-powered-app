import chainlit as cl
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.chains import LLMChain


@cl.on_chat_start
async def on_chat_start():
    model = ChatOpenAI(
        model='gpt-3.5-turbo',
        streaming=True
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are Chainlit GPT, a helpful assistant."
            ),
            (
                "human",
                "{question}"
            ),
        ]
    )

    chain = LLMChain(
        llm=model,
        prompt=prompt,
        output_parser=StrOutputParser()
    )

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    # load from user_session
    chain = cl.user_session.get("chain")

    response = await chain.arun(
        question=message.content,
        callbacks=[cl.LangchainCallbackHandler()]
    )

    await cl.Message(content=response).send()
