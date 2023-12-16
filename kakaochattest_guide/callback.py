import logging
import os
import time

import aiohttp
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.prompts import PromptTemplate

from dto import ChatbotRequest

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ.get('API_KEY')

SYSTEM_MSG = "당신은 카카오 서비스 제공자입니다."
logger = logging.getLogger("Callback")

import db.db

db.db.upload_kakao_social_data()
db.db.upload_kakao_sink_data()
db.db.upload_kakaotalk_channel_data()


def get_prompt(filename):
    with open("prompt/" + filename, "r") as fin:
        return fin.read()


llm = ChatOpenAI(temperature=0.1, model='gpt-3.5-turbo-16k')
system_message_prompt = SystemMessage(content='assistant는 user 가 카카오싱크에 대해 궁금한 부분을 천천히 친절하게 대답해줘.')
human = HumanMessagePromptTemplate.from_template('{text}\n---\n 에 대한 답변을 친절하게 대답해줘.')
prompt = ChatPromptTemplate.from_messages([system_message_prompt, human])

INTENT_PROMPT = PromptTemplate(
    template=get_prompt("intent_prompt.txt"),
    input_variables=['intent_list', 'context']
)

GUIDE_PROMPT = PromptTemplate(
    template=get_prompt("guide_prompt.txt"),
    input_variables=['related_doc', 'question', 'context']
)

FIND_INTENT_CHAIN = LLMChain(llm=llm, prompt=INTENT_PROMPT, verbose=True)
GUIDE_CHAIN = LLMChain(llm=llm, prompt=GUIDE_PROMPT, verbose=True)


async def callback_handler(request: ChatbotRequest) -> dict:
    input_text = request.userRequest.utterance

    file_path = os.path.join("history", "test.json")
    history = FileChatMessageHistory(file_path)
    context = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="user_message",
        chat_memory=history,
    ).buffer

    intent = FIND_INTENT_CHAIN.run(question=input_text, context=context)
    logger.info("intent: " + intent)

    if intent == "kakao_social":
        related_doc = db.db.query_on_kakao_social(input_text)
    elif intent == "kakao_sink":
        related_doc = db.db.query_on_kakao_sink(input_text)
    elif intent == "kakaotalk_channel":
        related_doc = db.db.query_on_kakaotalk_channel(input_text)
    else:
        print("TODO :: 매칭되는 INTENT 없음.")

    output_text = GUIDE_CHAIN.run(related_doc=related_doc, question=input_text, context=context)

    print(f"답변: {output_text}")

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
