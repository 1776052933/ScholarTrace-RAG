"""One-time, audited resume of the isolated 2026-10-21 LitSearch index."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from pymilvus import MilvusClient

from evaluation.current_datasets import load_dataset
from evaluation.m0_common import DATA, sha256, utc_now, write_json
from evaluation.run_current import code_digest


def main():
    directory = Path("/workspace/evaluation/runs/litsearch_full_v2_20261021")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_collection = "eval_current_64501a1afc8cf37e4bc22a34"
    assert manifest["dataset"] == "litsearch"
    assert manifest["settings"]["collection_name"] == expected_collection
    assert manifest["settings"]["milvus_uri"] == "http://milvus:19530"
    assert manifest["protocol"]["target_papers"] == 64183
    assert not (directory / "index_complete.json").exists()
    with sqlite3.connect(directory / "progress.sqlite3") as db:
        indexed = db.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    assert indexed == 23296, indexed
    dataset = load_dataset("litsearch", DATA)
    assert len(dataset.cases) == len(manifest["case_ids"]) == 597
    assert [case.case_id for case in dataset.cases] == manifest["case_ids"]
    for item, source in zip(manifest["files"], dataset.files):
        assert sha256(source) == item["sha256"]
    max_bytes = max(len(source.text.encode("utf-8")) for source in dataset.sources())
    assert 16384 < max_bytes <= 32768, max_bytes

    client = MilvusClient(uri="http://milvus:19530")
    description = client.describe_collection(expected_collection)
    fields = [field for field in description["fields"] if field["name"] == "text"]
    assert len(fields) == 1
    previous_limit = int(fields[0]["params"]["max_length"])
    assert previous_limit in (16384, 32768), previous_limit
    if previous_limit == 16384:
        client.alter_collection_field(
            collection_name=expected_collection,
            field_name="text",
            field_params={"max_length": 32768},
        )
    changed = client.describe_collection(expected_collection)
    fields = [field for field in changed["fields"] if field["name"] == "text"]
    assert int(fields[0]["params"]["max_length"]) == 32768
    assert changed["description"] == description["description"]

    old_digest = manifest["code_sha256"]
    new_digest = code_digest()
    assert old_digest != new_digest
    write_json(directory / "text_limit_migration.json", {
        "time": utc_now(),
        "reason": "Two released title/abstract records exceed the original 16 KB text field",
        "collection": expected_collection,
        "indexed_papers_before": indexed,
        "max_corpus_utf8_bytes": max_bytes,
        "old_text_limit": 16384,
        "new_text_limit": 32768,
        "old_manifest_sha256": sha256(manifest_path),
        "old_code_sha256": old_digest,
        "new_code_sha256": new_digest,
        "data_sha256_unchanged": True,
        "embedding_vectors_unchanged": True,
    })
    manifest["code_sha256"] = new_digest
    write_json(manifest_path, manifest)
    print(json.dumps({"indexed": indexed, "max_bytes": max_bytes,
                      "collection": expected_collection, "new_limit": 32768}))


if __name__ == "__main__":
    main()
