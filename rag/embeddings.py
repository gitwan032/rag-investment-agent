"""오픈소스 임베딩 모델(BAAI/bge-m3)을 LangChain 형식으로 불러온다."""
import os
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL


def pick_device() -> str:
    """Apple Silicon이면 mps(GPU), 아니면 cpu. EMBEDDING_DEVICE 환경변수로 강제 지정 가능."""
    forced = os.getenv("EMBEDDING_DEVICE")
    if forced:
        return forced
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """처음 호출할 때 한 번만 모델을 메모리에 올리고, 이후에는 같은 객체를 재사용한다."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": pick_device()},
        # 벡터 길이를 1로 맞춰 두면 코사인 유사도 = 내적 (수업 2-5 참고)
        encode_kwargs={"normalize_embeddings": True, "batch_size": 8},
    )
