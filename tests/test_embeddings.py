import httpx
import pytest

from opsdesk.embeddings import EmbeddingError, OllamaEmbedder



def install_response(monkeypatch, payload, status_code=200):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["body"] = kwargs["json"]

        request = httpx.Request(
            "POST",
            url,
            json=kwargs["json"],
        )

        return httpx.Response(
            status_code,
            json=payload,
            request=request,
        )

    monkeypatch.setattr(
        "opsdesk.embeddings.httpx.post",
        fake_post,
    )

    return captured

def test_sends_texts_and_returns_vectors(monkeypatch):
    captured = install_response(
        monkeypatch,
        {
            "embeddings": [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ]
        },
    )

    embedder = OllamaEmbedder()
    texts = ["上报程序没有及时收到确认", "配置格式不匹配"]

    vectors = embedder.embed_texts(texts)

    assert vectors == [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]
    assert captured["url"] == "http://127.0.0.1:11434/api/embed"
    assert captured["body"] == {
        "model": "bge-m3:567m",
        "input": texts,
        "truncate": False,
    }


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"embeddings": [[0.1, 0.2]]},
        {"embeddings": [[0.1, 0.2], [0.3]]},
        {"embeddings": [[0.0, 0.0], [0.1, 0.2]]},
    ],
)
def test_rejects_unusable_response(monkeypatch, payload):
    install_response(monkeypatch, payload)
    embedder = OllamaEmbedder()

    with pytest.raises(EmbeddingError):
        embedder.embed_texts(["第一条文本", "第二条文本"])

def test_reports_http_failure(monkeypatch):
    install_response(
        monkeypatch,
        {"error": "model not found"},
        status_code=404,
    )

    with pytest.raises(EmbeddingError, match="HTTP 404"):
        OllamaEmbedder().embed_texts(["设备没有收到确认"])


def test_rejects_blank_text_before_request():
    with pytest.raises(ValueError, match="非空文本"):
        OllamaEmbedder().embed_texts(["   "])