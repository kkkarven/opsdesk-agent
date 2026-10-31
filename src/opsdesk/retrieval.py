"""在适用资料中执行向量相似度检索。"""

import json
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile

import faiss
import numpy as np

from opsdesk.embeddings import OllamaEmbedder
from opsdesk.knowledge import KnowledgeChunk, select_applicable

@dataclass(frozen=True)
class SearchHit:
    score:float
    chunk:KnowledgeChunk


def normalize_vectors(values:list[list[float]]) -> np.ndarray:
    matrix = np.array(
        values,
        dtype=np.float32,
        order="C",
        copy = True
    )

    if(
        matrix.ndim !=2
        or matrix.shape[0] == 0
        or matrix.shape[1] == 0
    ):
        raise ValueError("向量必须组成非空二维数组")

    if not np.isfinite(matrix).all():
        raise ValueError("向量包含不可用的数值")

    lengths = np.linalg.norm(
        matrix.astype(np.float64),
        axis=1,
        keepdims=True,
    )

    if (lengths == 0).any():
        raise ValueError("不能使用全零向量")

    normalized = matrix /lengths

    return np.ascontiguousarray(
        normalized,
        dtype=np.float32,
    )

class VectorRetriever:
    def __init__(
        self,
        chunks:list[KnowledgeChunk],
        vectors:list[list[float]],
        embedding_model:str,
    ):
        if not chunks:
            raise ValueError("资料片段不能为空")

        chunk_ids = [chunk.chunk_id for chunk in chunks]

        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("资料中存在重复的 chunk_id")

        matrix = normalize_vectors(vectors)

        if matrix.shape[0] != len(chunks):
            raise ValueError("向量数量与资料片段数量不一致")

        self.embedding_model = embedding_model
        self.dimension = matrix.shape[1]

        self._chunks = [
            chunk.model_copy(deep=True)
            for chunk in chunks
        ]
        self._vectors = matrix

        self._row_by_id = {
            chunk.chunk_id:row
            for row,chunk in enumerate(self._chunks)
        }

    @classmethod
    def from_chunks(
        cls,
        chunks:list[KnowledgeChunk],
        embedder: OllamaEmbedder,
    ) -> "VectorRetriever":
        if not chunks:
            raise ValueError("资料片段不能为空")

        texts = [chunk.text for chunk in chunks]
        vectors = embedder.embed_texts(texts)

        return cls(
            chunks = chunks,
            vectors = vectors,
            embedding_model = embedder.model
        )

    def search(
            self,
            question:str,
            device_model:str|None,
            at:datetime,
            embedder:OllamaEmbedder,
            top_k:int = 3,
    ) -> list[SearchHit]:
        if not isinstance(question,str) or not question.strip():
            raise ValueError("问题不能为空")

        if (
            isinstance(top_k,bool)
            or not isinstance(top_k,int)
            or top_k <= 0
        ):
            raise ValueError("top_k 必须是正整数")

        if embedder.model != self.embedding_model:
            raise ValueError("查询模型与建立文档向量的模型不一致")

        applicable = select_applicable(
            self._chunks,
            model=device_model,
            at = at,
        )

        if not applicable:
            return []

        rows = [
            self._row_by_id[chunk.chunk_id]
            for chunk in applicable
        ]

        candidate_vectors = np.ascontiguousarray(
            self._vectors[rows],
            dtype=np.float32,
        )

        query_vectors = normalize_vectors(
            embedder.embed_texts([question])
        )

        if query_vectors.shape != (1,self.dimension):
            raise ValueError("查询向量数量或维度与索引不匹配")

        index = faiss.IndexFlatIP(self.dimension)
        index.add(candidate_vectors)

        result_count = min(top_k,len(applicable))
        scores,positions = index.search(
            query_vectors,
            result_count,
        )

        hits = []

        for score,position in zip(
            scores[0],
            positions[0],
            strict=True,
        ):
            chunk = applicable[int(position)]

            hits.append(
                SearchHit(
                    score=float(score),
                    chunk=chunk.model_copy(deep=True)
                )
            )
        return hits

    def save(self,path:Path) ->None:
        metadata = {
            "format_version":1,
            "embedding_model":self.embedding_model,
            "dimension":self.dimension,
            "chunk_count":len(self._chunks),
            "metric":"cosine",
            "normalization":"l2",
            "document_input":"chunk.text",
            "chunks":[
                chunk.model_dump(mode="json")
                for chunk in self._chunks
            ],
        }

        metadata_text = json.dumps(
            metadata,
            ensure_ascii=False,
            sort_keys=True,                           
        )

        checksum = sha256(
            metadata_text.encode("utf-8")
            + b"\0" + self._vectors.tobytes(order="C")
        ).hexdigest()

        path.parent.mkdir(parents=True,exist_ok=True)
        temporary_path = None

        try:
            with NamedTemporaryFile(
                mode = "wb",
                dir = path.parent,
                prefix = ".snapshot-",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)

                np.savez(
                    stream,
                    metadata = np.array(metadata_text),
                    vectors = self._vectors,
                    checksum = np.array(checksum),
                    allow_pickle=False,
                )

            temporary_path.replace(path)

        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @classmethod
    def load(
        cls,
        path:Path,
        expected_model:str,
    ) -> "VectorRetriever":
        with np.load(path,allow_pickle=False) as saved:
            if set(saved.files) != {
                "metadata",
                "vectors",
                "checksum",
            }:
                raise ValueError("快照文件内容不完整或格式不支持")

            metadata_text = saved["metadata"].item()
            vectors = saved["vectors"].copy(order="C")
            stored_checksum = saved["checksum"].item()

        if not isinstance(metadata_text,str):
            raise ValueError("快照元数据必须是 JSON 文本")

        if not isinstance(stored_checksum,str):
            raise ValueError("快照缺少合法的校验值")

        if vectors.ndim != 2 or vectors.dtype != np.dtype("float32"):
            raise ValueError("快照向量必须是 float32 二维数组")

        actual_checksum = sha256(
            metadata_text.encode("utf-8")
            + b"\0"
            + vectors.tobytes(order="C")
        ).hexdigest()

        if actual_checksum != stored_checksum:
            raise ValueError("快照完整性校验失败")

        metadata = json.loads(metadata_text)

        if not isinstance(metadata,dict):
            raise ValueError("快照元数据必须是对象")

        if metadata.get("format_version") != 1:
            raise ValueError("不支持这个快照格式版本")

        if (
            metadata.get("metric") != "cosine"
            or metadata.get("normalization") != "l2"
            or metadata.get("document_input") != "chunk.text"
        ):
            raise ValueError("快照的向量处理配置不匹配")

        stored_model = metadata.get("embedding_model")

        if not isinstance(stored_model,str) or not stored_model:
            raise ValueError("快照缺少合法的模型名称")

        if stored_model != expected_model:
            raise ValueError("快照模型与当前配置不一致")

        expected_shape = (
            metadata.get("chunk_count"),
            metadata.get("dimension"),
        )

        if vectors.shape != expected_shape:
            raise ValueError("快照向量形状与元数据不一致")

        chunk_data = metadata.get("chunks")

        if not isinstance(chunk_data,list):
            raise ValueError("快照缺少资料片段列表")

        chunks = [
            KnowledgeChunk.model_validate(item)
            for item in chunk_data
        ]

        return cls(
            chunks = chunks,
            vectors = vectors.tolist(),
            embedding_model = stored_model,
        )