import json

import chromadb
from chromadb.api.models import Collection
from langchain_community.document_loaders import TextLoader

CHROMA_PERSIST_PATH = "chroma"
KAKAO_SOCIAL_COLLECTION_NAME = "kakao-social-collection"
KAKAO_SINK_COLLECTION_NAME = "kakao-sink-collection"
KAKAOTALK_CHANNEL_COLLECTION_NAME = "kakaotalk-channel-collection"


def _createCollection(collection_name: str):
    return chromadb.PersistentClient().get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )


kakao_social_collection = _createCollection(KAKAO_SOCIAL_COLLECTION_NAME)
kakao_sink_collection = _createCollection(KAKAO_SINK_COLLECTION_NAME)
kakaotalk_channel_collection = _createCollection(KAKAOTALK_CHANNEL_COLLECTION_NAME)


def _text_to_json(file_path: str):
    loader = {
        "txt": TextLoader,
    }.get(file_path.split(".")[-1])

    if loader is None:
        raise ValueError("Not supported file type")

    f = open(file_path, "r")
    full_txt = f.read()

    json_file = []
    for txt in full_txt.split('\n#'):
        t = [i for i in txt.split('\n') if i != '']
        if len(t) <= 1:
            continue
        json_file.append({
            "Title": t[0],
            "Description": ''.join(t[1:])
        })

    res = json.dumps(json_file, ensure_ascii=False)
    return json.loads(res)


def upload_kakao_social_data():
    print("카카오소셜 chroma에 업로드...")
    file_path = './data/카카오소셜.txt'
    data = _text_to_json(file_path=file_path)
    _upload(kakao_social_collection, data)


def upload_kakao_sink_data():
    print("카카오싱크 chroma에 업로드...")
    file_path = './data/카카오싱크.txt'
    data = _text_to_json(file_path=file_path)
    _upload(kakao_sink_collection, data)


def upload_kakaotalk_channel_data():
    print("카카오톡 채널 chroma에 업로드...")
    file_path = './data/카카오톡채널.txt'
    data = _text_to_json(file_path=file_path)
    _upload(kakaotalk_channel_collection, data)


def _upload(collection: Collection, data: json):
    # 데이터 준비
    ids = []  # 인덱스
    documents = []  # 벡터로 변환 저장할 텍스트 데이터로 ChromaDB에 Embedding 데이터가 없으면 자동으로 벡터로 변환해서 저장

    for item in data:
        id = item['Title'].replace(' ', '-')
        document = f"{item['Title'].strip()} : {item['Description'].strip()}"

        ids.append(id)
        documents.append(document)

    collection.add(
        documents=documents,
        ids=ids
    )


def query_on_kakao_social(query: str) -> list[str]:
    return _query_db(kakao_social_collection, query)


def query_on_kakao_sink(query: str) -> list[str]:
    return _query_db(kakao_sink_collection, query)


def query_on_kakaotalk_channel(query: str) -> list[str]:
    return _query_db(kakaotalk_channel_collection, query)


def _query_db(collection: Collection, query: str) -> list[str]:
    docs = collection.query(  # collection.similarity_search(query)
        query_texts=[query],
        n_results=3,
    )
    print(f"vector search result: {docs['documents'][0]}")
    return docs["documents"][0]
