import logging
import os
import time

import aiohttp
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.prompts import PromptTemplate

import db.db
from dto import ChatbotRequest

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ.get('API_KEY')

logger = logging.getLogger("Callback")

db.db.upload_kakao_social_data()
db.db.upload_kakao_sink_data()
db.db.upload_kakaotalk_channel_data()


def get_prompt(filename):
    with open("prompt/" + filename, "r") as fin:
        return fin.read()


llm = ChatOpenAI(temperature=0.1, model='gpt-3.5-turbo-16k')

INTENT_PROMPT = PromptTemplate(
    template=get_prompt("intent_prompt.txt"),
    input_variables=['intent_list', 'context']
)

GUIDE_PROMPT = PromptTemplate(
    template=get_prompt("guide_prompt.txt"),
    input_variables=['related_doc', 'question', 'context']
)

INTENT_NOT_FOUND_PROMPT = PromptTemplate(
    template=get_prompt("intent_not_found_prompt.txt"),
    input_variables=['question']
)

FIND_INTENT_CHAIN = LLMChain(llm=llm, prompt=INTENT_PROMPT, verbose=True)
GUIDE_CHAIN = LLMChain(llm=llm, prompt=GUIDE_PROMPT, verbose=True)
INTENT_NOT_FOUND_CHAIN = LLMChain(llm=llm, prompt=INTENT_NOT_FOUND_PROMPT, verbose=True)


async def callback_handler(request: ChatbotRequest) -> dict:
    input_text = request.userRequest.utterance

    file_path = os.path.join("history", "test.json")
    history = FileChatMessageHistory(file_path)
    context = ConversationBufferWindowMemory(
        k=3,
        memory_key="chat_history",
        input_key="user_message",
        chat_memory=history,
    ).buffer

    intent = FIND_INTENT_CHAIN.run(question=input_text, context=context)
    logger.info("intent: " + intent)

    if intent == "kakao_social":
        related_doc = db.db.query_on_kakao_social(input_text)
        output_text = GUIDE_CHAIN.run(related_doc=related_doc, question=input_text, context=context)
    elif intent == "kakao_sink":
        related_doc = db.db.query_on_kakao_sink(input_text)
        output_text = GUIDE_CHAIN.run(related_doc=related_doc, question=input_text, context=context)
    elif intent == "kakaotalk_channel":
        related_doc = db.db.query_on_kakaotalk_channel(input_text)
        output_text = GUIDE_CHAIN.run(related_doc=related_doc, question=input_text, context=context)
    else:
        output_text = INTENT_NOT_FOUND_CHAIN.run(question=input_text)

    logger.info(f"답변: {output_text}")

    history.add_user_message(input_text)
    history.add_ai_message(output_text)

    payload = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": output_text
                    }
                }
            ]
        }
    }
    # ===================== end =================================
    # 참고링크1 : https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/ai_chatbot_callback_guide
    # 참고링크1 : https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/answer_json_format

    time.sleep(1.0)

    url = request.userRequest.callbackUrl

    if url:
        async with aiohttp.ClientSession() as session:
            async with session.post(url=url, json=payload, ssl=False) as resp:
                await resp.json()
