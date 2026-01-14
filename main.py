from __future__ import annotations

from src.docsim.compare import compare_documents
from src.docsim.report_formatter import format_report


def load_text(path: str) -> str:
    """reading .txt files"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":

    ref_path = "reference.txt"
    gen_path = "generated.txt"

    print("Document similarity check\n")
    print(f"Reference: {ref_path}")
    print(f"Generated: {gen_path}\n")

    ref_text = load_text(ref_path)
    gen_text = load_text(gen_path)

    result = compare_documents(ref_text, gen_text)
    report = format_report(result)

    print(report)

    print("\nMetrics detailed")
    print("Cosine similarity:", result["scores"]["cosine"])
    print("ROUGE-L F1:       ", result["scores"]["rougeL_f1"])
    print("BERTScore F1:     ", result["scores"]["bertscore_f1"])
    print("Numeric Jaccard:  ", result["scores"]["numeric_jaccard"])
    print("Textual index:    ", result["scores"]["textual_index"])
    print("Overall index:    ", result["scores"]["overall_index"])

    # chunk info
    dbg = result.get("debug", {})
    if dbg:
        print("\nChunk debug")
        print("Ref chunks:", dbg.get("ref_chunks"))
        print("Gen chunks:", dbg.get("gen_chunks"))
        print("Chunk words:", dbg.get("chunk_words"))
        print("Length penalty:", dbg.get("length_penalty"))

    print("\nFinished.")
