from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

from src.docsim.compare import compare_documents


app = FastAPI(
    title="DocSim Similarity API",
    description=(
        "Comparing the reference and generated text. "
        "Returns the metrics and composite indexes."
    ),
    version="0.1.0",
)


class SimilarityScoresResponse(BaseModel):
    """
    Response JSON.
    """
    cosine_similarity: float
    rougeL_f1: float
    bertscore_f1: float
    numeric_jaccard: float
    textual_index: float
    overall_index: float


@app.get("/health")
def health() -> Dict[str, str]:
    """Health check"""
    return {"status": "ok"}


@app.post("/compare-files", response_model=SimilarityScoresResponse)
async def compare_files(
    reference: UploadFile = File(..., description="Reference report .txt format"),
    generated: UploadFile = File(..., description="Generated report .txt format"),
) -> SimilarityScoresResponse:
    """
    Comparing the inputs
    """

    # Basic type check
    for f, name in ((reference, "reference"), (generated, "generated")):
        if f.content_type not in ("text/plain", "application/octet-stream"):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"A(z) '{name}' Not a text (text/plain). "
                    f"Recieved content_type: {f.content_type}"
                ),
            )

    try:
        ref_bytes = await reference.read()
        gen_bytes = await generated.read()
        ref_text = ref_bytes.decode("utf-8")
        gen_text = gen_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Can't read the input as an UTF-8 coded file. "
                   "Please use UTF-8 format.",
        )

    # main comparison
    result: Dict[str, Any] = compare_documents(ref_text, gen_text)
    scores: Dict[str, Any] = result["scores"]

    def r3(x: float) -> float:
        """return as a float rounded by 3 digits."""
        return float(f"{x:.3f}")

    response = SimilarityScoresResponse(
        cosine_similarity=r3(scores["cosine"]),
        rougeL_f1=r3(scores["rougeL_f1"]),
        bertscore_f1=r3(scores["bertscore_f1"]),
        numeric_jaccard=r3(scores["numeric_jaccard"]),
        textual_index=r3(scores["textual_index"]),
        overall_index=r3(scores["overall_index"]),
    )

    return response
