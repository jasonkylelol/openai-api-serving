from openai import OpenAI
from datetime import datetime
import os

api_key = "EMPTY"
model = "embedding-3"
server_addr = os.getenv("SERVER_ADDR")

def init_client():
    client = OpenAI(
        api_key=api_key,
        base_url=f"http://{server_addr}/v1",
    )
    return client


def xprint(msg):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


if __name__ == "__main__":
    xprint("start")
    stime = datetime.now()

    client = init_client()
    responses = client.embeddings.create(
        input=[
            "你有这么高速运转的机械进入中国，记住我给出的原理！",
        ],
        model=model)
    for data in responses.data:
        print(data.embedding)

    xprint(f"finished cost: {datetime.now() - stime}")
