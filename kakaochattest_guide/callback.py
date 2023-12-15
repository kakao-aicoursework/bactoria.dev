import logging
import os
import time

import aiohttp
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage

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

INTENT_PROMPT = get_prompt("intent_prompt.txt")
FIND_INTENT_CHAIN = LLMChain(llm=llm, prompt=ChatPromptTemplate.from_messages([INTENT_PROMPT]), verbose=True)
chain = LLMChain(llm=llm, prompt=prompt, verbose=True)


async def callback_handler(request: ChatbotRequest) -> dict:
    input_text = request.userRequest.utterance

    intent = FIND_INTENT_CHAIN.run(input_text)
    logger.info("intent: " + intent)

    if intent == "kakao_social":
        related_docs = db.db.query_on_kakao_social(input_text)
    elif intent == "kakao_sink":
        related_docs = db.db.query_on_kakao_sink(input_text)
    elif intent == "kakaotalk_channel":
        related_docs = db.db.query_on_kakaotalk_channel(input_text)
    else:
        output_text = "대답할 수 없어요."

    print(f"related_docs: {related_docs}")
    output_text = related_docs

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
