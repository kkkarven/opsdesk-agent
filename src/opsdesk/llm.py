"""调用 Ollama 的本地聊天模型。"""

import httpx

class GenerationError(RuntimeError):
    pass

class OllamaChat:
    def __init__(
        self,
        model:str = "qwen3.5:2b-q4_K_M",
        base_url:str = "http://127.0.0.1:11434",
        timeout:float = 180.0
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(
            self,
            messages:list[dict[str,str]],
            schema:dict,
    ) -> str:
        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json={
                    "model":self.model,
                    "messages":messages,
                    "format":schema,
                    "stream":False,
                    "think":False,
                    "options":{
                        "temperature":0,
                        "num_ctx":8192,
                        "num_predict":1024,
                    },
                },
                timeout=httpx.Timeout(
                    self.timeout,
                    connect=10.0,
                ),
                trust_env=False,
            )
            response.raise_for_status()

        except httpx.HTTPStatusError as error:
            raise GenerationError(
                f"Ollama 返回 HTTP {error.response.status_code}："
                f"{error.response.text}"
            ) from error

        except httpx.RequestError as error:
            raise GenerationError(
                f"无法完成Ollama请求“{error}"
            ) from error

        try:
            payload = response.json()
        except ValueError as error:
            raise GenerationError(
                "Ollama 响应不是合法 JSON"
            )from error

        if not isinstance(payload,dict):
            raise GenerationError("Ollama响应必须是对象")

        if payload.get("done") is not True:
            raise GenerationError("Ollama没有返回完整回答")

        if payload.get("done_reason") == "length":
            raise GenerationError("模型回答达到长度限制，未完整生成")

        message = payload.get("message")
        if not isinstance(message,dict):
            raise GenerationError("Ollama响应缺少message")

        content = message.get("content")
        if not isinstance(content,str) or not content.strip():
            raise GenerationError("Ollama返回了空回答")

        return content