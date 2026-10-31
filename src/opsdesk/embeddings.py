"""通过本地 Ollama 将文本转换为向量。"""

from math import isfinite

import httpx

class EmbeddingError(RuntimeError):
    pass

class OllamaEmbedder:
    def __init__(
        self,
        model:str = "bge-m3:567m",
        base_url = "http://127.0.0.1:11434",
        timeout: float = 180.0,             
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def embed_texts(
            self,
            texts:list[str],
    ) -> list[list[float]]:
        if not isinstance(texts,list) or not texts:
            raise ValueError("texts 必须是非空列表")

        for text in texts:
            if not isinstance(text,str) or not text.strip():
                raise ValueError("每条输入必须是非空文本")

        try:
            response = httpx.post(
                f"{self.base_url}/api/embed",
                json={
                    "model":self.model,
                    "input":texts,
                    "truncate":False,
                },
            timeout = httpx.Timeout(
                self.timeout,
                connect=10.0,
            ),
            trust_env = False,
        )
            response.raise_for_status()

        except httpx.HTTPStatusError as error:
            raise EmbeddingError(
                f"Ollama 返回 HTTP {error.response.status_code}："
                f"{error.response.text}"
            ) from error

        except httpx.RequestError as error:
            raise EmbeddingError(
                f"无法完成 Ollama Embedding 请求：{error}"
            ) from error

        try:
            payload = response.json()
        except ValueError as error:
            raise EmbeddingError("Ollama 返回的内容不是合法 JSON") from error

        if not isinstance(payload,dict):
            raise EmbeddingError("Ollama 返回的 JSON 必须是对象")

        vectors = payload.get("embeddings")

        if not isinstance(vectors,list):
            raise EmbeddingError("返回结果缺少 embeddings 列表")

        if len(vectors) != len(texts):
            raise EmbeddingError("返回向量数量与输入文本数量不一致")

        dimension = None
        result = []

        for position,vector in enumerate(vectors):
            if not isinstance(vector,list) or not vector:
                raise EmbeddingError(
                    f"第 {position + 1} 个向量必须是非空列表"
                )

            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise EmbeddingError("返回向量的维度不一致")

            checked_vector = []

            for value in vector:
                if (isinstance(value,bool) or not isinstance(value,(int,float))):
                    raise EmbeddingError("向量元素必须是数值")

                try:
                    number = float(value)

                except OverflowError as error:
                    raise EmbeddingError("向量元素超出可用数值范围") from error

                if not isfinite(number):
                    raise EmbeddingError("向量包含NaN或为无穷大")

                checked_vector.append(number)

            if all(value == 0.0 for value in checked_vector):
                raise EmbeddingError("模型返回了全零向量")

            result.append(checked_vector)

        return result