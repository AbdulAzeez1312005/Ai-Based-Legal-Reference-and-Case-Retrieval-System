"""Offline regression tests for generic provision-aware parsing and retrieval."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import legal_rag


class FakeEmbedder:
    def encode(self, texts, **kwargs):
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeIndex:
    def __init__(self, matches):
        self.matches = matches
        self.calls = []

    @staticmethod
    def matches_filter(metadata, filter_value):
        if not filter_value:
            return True
        if "$and" in filter_value:
            return all(FakeIndex.matches_filter(metadata, part)
                       for part in filter_value["$and"])
        for key, condition in filter_value.items():
            value = metadata.get(key)
            if isinstance(condition, dict):
                if "$eq" in condition and value != condition["$eq"]:
                    return False
                if "$gte" in condition and (value is None or value < condition["$gte"]):
                    return False
                if "$lte" in condition and (value is None or value > condition["$lte"]):
                    return False
            elif value != condition:
                return False
        return True

    def query(self, **kwargs):
        self.calls.append(kwargs)
        matches = [item for item in self.matches
                   if self.matches_filter(item["metadata"], kwargs.get("filter"))]
        matches.sort(key=lambda item: item["score"], reverse=True)
        return {"matches": matches[:kwargs["top_k"]]}


def make_match(source, number, provision_id, chunk_index, text, score=0.9):
    return {
        "id": f"{source}-{number}-{chunk_index}", "score": score,
        "metadata": {
            "text": text, "source": source, "document_id": source,
            "document_type": "statute", "page": 12 + chunk_index,
            "chunk_index": chunk_index, "provision_id": provision_id,
            "provision_type": "section", "provision_number": number,
            "provision_title": "Illustrative rule", "ingest_version": legal_rag.INGEST_VERSION,
        },
    }


class ProvisionParsingTests(unittest.TestCase):
    def test_file_classifier_uses_generic_filename_cues(self):
        self.assertEqual(legal_rag.detect_document_type(
            "THE CODE OF CIVIL PROCEDURE, 1908.pdf"), "statute")
        self.assertEqual(legal_rag.detect_document_type(
            "THE RIGHT TO INFORMATION ACT, 2005.pdf",
            "The Constitution of India is referred to in this Act."), "statute")
        self.assertEqual(legal_rag.detect_document_type(
            "THE CONSTITUTION OF INDIA.pdf"), "constitution")
        self.assertEqual(legal_rag.detect_document_type(
            "A v B judgment.pdf"), "case")
        self.assertEqual(legal_rag.detect_document_type(
            "indian_laws_definitions.txt"), "reference")
    def test_extracts_section_article_and_subsection(self):
        self.assertEqual(
            legal_rag.extract_provision_references(
                "Compare Section 10A(2) and Article 21."
            ),
            [
                {"type": "section", "number": "10A", "subsection": "2"},
                {"type": "article", "number": "21", "subsection": ""},
            ],
        )

    def test_article_question_does_not_parse_the_word_in_as_rule_i(self):
        self.assertEqual(
            legal_rag.extract_provision_references("What is the rule in Article 21?"),
            [{"type": "article", "number": "21", "subsection": ""}],
        )

    def test_extracts_a_list_of_sections(self):
        refs = legal_rag.extract_provision_references(
            "How do Sections 10, 11 and 12 differ?"
        )
        self.assertEqual([row["number"] for row in refs], ["10", "11", "12"])

    def test_discovers_headings_without_act_specific_rules(self):
        text = (
            "Section 1. Short title\nThis provision defines the name and scope of this law.\n"
            "Section 2. Definitions\nThis provision explains the terms used throughout the law.\n"
            "Article 21. Protection of life\nEvery person is entitled to protection under law."
        )
        found = legal_rag.find_provision_headings(text, "statute")
        self.assertEqual(
            [(item["type"], item["number"]) for item in found],
            [("section", "1"), ("section", "2"), ("article", "21")],
        )

    def test_constitution_articles_are_distinct_from_schedule_paragraphs(self):
        schedule_body = "General schedule text. " * 100
        text = (
            "FIRST SCHEDULE\nSECOND SCHEDULE\nA short contents list.\n"
            "Article 20. Existing protection.\nThe main constitutional article applies.\n"
            + ("Intervening constitutional provisions. " * 120)
            + "\nSCHEDULE II\n" + schedule_body
            + "\n21. Autonomous districts and autonomous regions.\n"
        )
        found = legal_rag.find_provision_headings(text, "constitution")
        article = next(item for item in found
                       if item["type"] == "article" and item["number"] == "20")
        paragraph = next(item for item in found
                         if item["type"] == "paragraph" and item["number"] == "21")
        self.assertEqual(paragraph["context"], "SCHEDULE II")
        self.assertNotIn("article", [item["type"] for item in found
                                    if item["number"] == "21"])

    def test_real_constitution_pdf_has_article_text_and_schedule_paragraphs(self):
        path = (
            Path(__file__).resolve().parent / "data2"
            / "THE CONSTITUTION OF INDIA.pdf"
        )
        records = legal_rag.records_for_file(path)
        for number, expected in (
            ("14", "equality before the law"),
            ("19", "freedom of speech and expression"),
            ("21", "deprived of his life or personal liberty"),
        ):
            article = [
                row for row in records
                if row["metadata"].get("provision_type") == "article"
                and row["metadata"].get("provision_number") == number
            ]
            self.assertTrue(article, f"Article {number} was not parsed")
            text = " ".join(row["text"] for row in article).lower()
            self.assertIn(expected, text)
        article_21 = " ".join(
            row["text"] for row in records
            if row["metadata"].get("provision_type") == "article"
            and row["metadata"].get("provision_number") == "21"
        )
        self.assertNotIn("Right to education", article_21)
        schedule_paragraphs = [
            row for row in records
            if row["metadata"].get("provision_type") == "paragraph"
            and row["metadata"].get("provision_number") == "21"
        ]
        self.assertTrue(schedule_paragraphs)
        self.assertTrue(all(row["metadata"].get("provision_context", "").endswith("SCHEDULE")
                            for row in schedule_paragraphs))

    def test_discovers_bare_numbered_statute_headings(self):
        text = (
            "1. Short title\nThe text of the first provision is present here and long enough.\n"
            "2. Definitions\nThe second provision has explanatory wording."
        )
        found = legal_rag.find_provision_headings(text, "statute")
        self.assertEqual([item["number"] for item in found], ["1", "2"])

    def test_case_documents_are_not_split_on_numbered_paragraphs(self):
        text = "1. First judgment paragraph with useful text.\n2. This is the next paragraph."
        self.assertEqual(legal_rag.find_provision_headings(text, "case"), [])

    def test_chunking_obeys_limit_and_overlap(self):
        value = ("A long legal sentence with enough words. " * 90).strip()
        chunks = list(legal_rag._chunk_text(value, size=300, overlap=50))
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 300 for _, chunk in chunks))
        self.assertLess(chunks[1][0], chunks[0][0] + len(chunks[0][1]))


class GenericRetrievalTests(unittest.TestCase):
    def test_exact_section_reassembles_every_matching_chunk(self):
        first = make_match(
            "General Act.pdf", "7", "general:section:7", 0,
            "Section 7. A person must attend the hearing and comply with ",
        )
        second = make_match(
            "General Act.pdf", "7", "general:section:7", 1,
            "attend the hearing and comply with the written order.",
        )
        unrelated = make_match(
            "Other Act.pdf", "7", "other:section:7", 0,
            "Section 7. A separate topic with unrelated material.", 0.3,
        )
        index = FakeIndex([first, second, unrelated])
        results, trace = legal_rag.retrieve_legal_context(
            "What does Section 7 say about attending a hearing?",
            index, FakeEmbedder(), top_k=4,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["metadata"]["source"], "General Act.pdf")
        self.assertEqual(results[0]["chunks"], 2)
        self.assertIn("written order", results[0]["page_content"])
        self.assertEqual(trace[0]["mode"], "complete-provision")

    def test_reassembled_provision_tracks_all_source_pages(self):
        first = make_match(
            "Multi-page Act.pdf", "7", "multi:section:7", 0,
            "Section 7. First page of the legal provision.",
        )
        second = make_match(
            "Multi-page Act.pdf", "7", "multi:section:7", 1,
            "Second page continues the same legal provision.",
        )
        results, _trace = legal_rag.retrieve_legal_context(
            "What does Section 7 say about this legal provision?",
            FakeIndex([first, second]), FakeEmbedder(),
        )
        self.assertEqual(results[0]["metadata"]["pages"], [12, 13])

    def test_named_crpc_filters_ambiguous_section_number(self):
        crpc = make_match(
            "THE CODE OF CRIMINAL PROCEDURE, 1973.pdf", "437",
            "crpc:section:437", 0,
            "Section 437. The court may release an accused person on bail subject to "
            "conditions, including attendance and cooperation.",
            0.55,
        )
        companies = make_match(
            "THE COMPANIES ACT, 2013.pdf", "437",
            "companies:section:437", 0,
            "Section 437. The Tribunal may impose penalties for contraventions.",
            0.98,
        )
        index = FakeIndex([crpc, companies])
        results, _trace = legal_rag.retrieve_legal_context(
            "According to CrPC Section 437, what bail conditions may the court impose?",
            index, FakeEmbedder(),
        )
        self.assertEqual(
            [row["metadata"]["source"] for row in results],
            ["THE CODE OF CRIMINAL PROCEDURE, 1973.pdf"],
        )

    def test_context_free_duplicate_section_is_reported_as_ambiguous(self):
        matches = [
            make_match(
                "THE CODE OF CRIMINAL PROCEDURE, 1973.pdf", "437",
                "crpc:section:437", 0, "Section 437. Bail provisions.", 0.9,
            ),
            make_match(
                "THE COMPANIES ACT, 2013.pdf", "437",
                "companies:section:437", 0, "Section 437. Tribunal provisions.", 0.8,
            ),
        ]
        index = FakeIndex(matches)
        results, trace = legal_rag.retrieve_legal_context(
            "What is Section 437?", index, FakeEmbedder()
        )
        self.assertEqual(results, [])
        self.assertEqual(trace[0]["mode"], "ambiguous-provision")
        self.assertEqual(
            set(trace[0]["sources"]),
            {"THE CODE OF CRIMINAL PROCEDURE, 1973.pdf", "THE COMPANIES ACT, 2013.pdf"},
        )

    def test_unnamed_article_prefers_constitution_over_same_numbered_act_provision(self):
        constitution = make_match(
            "THE CONSTITUTION OF INDIA.pdf", "21",
            "constitution:article:21", 0,
            "Article 21. No person shall be deprived of life or personal liberty.",
            0.7,
        )
        constitution["metadata"].update({
            "provision_type": "article", "document_type": "constitution",
        })
        other_act = make_match(
            "THE RIGHT TO INFORMATION ACT, 2005.pdf", "21",
            "rti:article:21", 0,
            "Article 21. Protection of action taken in good faith.",
            0.99,
        )
        other_act["metadata"]["provision_type"] = "article"
        other_act["metadata"]["document_type"] = "constitution"
        results, _trace = legal_rag.retrieve_legal_context(
            "What is the rule in Article 21?",
            FakeIndex([other_act, constitution]), FakeEmbedder(),
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["metadata"]["source"], "THE CONSTITUTION OF INDIA.pdf")

    def test_context_ranks_ambiguous_section_by_its_subject(self):
        crpc = make_match(
            "THE CODE OF CRIMINAL PROCEDURE, 1973.pdf", "437",
            "crpc:section:437", 0,
            "Section 437. The court may release an accused person on bail subject to "
            "conditions, including attendance and cooperation.",
            0.55,
        )
        companies = make_match(
            "THE COMPANIES ACT, 2013.pdf", "437",
            "companies:section:437", 0,
            "Section 437. The Tribunal may impose conditions for registration.",
            0.98,
        )
        results, trace = legal_rag.retrieve_legal_context(
            "According to Section 437, what bail conditions may a court impose?",
            FakeIndex([crpc, companies]), FakeEmbedder(),
        )
        self.assertEqual(
            results[0]["metadata"]["source"],
            "THE CODE OF CRIMINAL PROCEDURE, 1973.pdf",
        )
        self.assertEqual(trace[0]["source"], "THE CODE OF CRIMINAL PROCEDURE, 1973.pdf")

    def test_article_query_uses_article_metadata(self):
        article = make_match(
            "Constitution.pdf", "21", "constitution:article:21", 0,
            "Article 21. Protection of life and personal liberty.",
        )
        article["metadata"]["provision_type"] = "article"
        index = FakeIndex([article])
        results, _ = legal_rag.retrieve_legal_context(
            "Explain Article 21.", index, FakeEmbedder()
        )
        self.assertEqual(results[0]["metadata"]["provision_type"], "article")

    def test_semantic_search_falls_back_to_legacy_records(self):
        old = {
            "id": "old", "score": 0.7,
            "metadata": {"text": "A case discusses the legal issue.", "source": "Case.pdf",
                         "page": 2, "chunk_index": 0},
        }
        index = FakeIndex([old])
        results, trace = legal_rag.retrieve_legal_context(
            "What did the judgment decide?", index, FakeEmbedder(), top_k=2
        )
        self.assertTrue(results)
        self.assertEqual(trace[0]["mode"], "semantic-window")
        self.assertTrue(any(call.get("filter") == {"ingest_version": legal_rag.INGEST_VERSION}
                            for call in index.calls))
        self.assertTrue(all(
            not isinstance(condition, dict)
            for call in index.calls
            for condition in call.get("filter", {}).values()
        ))

    def test_named_case_is_ranked_ahead_of_semantically_similar_statute(self):
        judgment = make_match(
            "Suprme-Court-Judgement_-Kesavananda-Bharati-v-State-of-Kerala-1973.pdf",
            "0", "kesavananda:document", 0,
            "Kesavananda Bharati held that Parliament's amending power is limited by "
            "the basic structure doctrine.",
            0.65,
        )
        judgment["metadata"].pop("provision_type")
        judgment["metadata"].pop("provision_number")
        statute = make_match(
            "THE CONSTITUTION OF INDIA.pdf", "0", "constitution:document", 0,
            "The Constitution describes limits on amending power and fundamental rights.",
            0.95,
        )
        statute["metadata"].pop("provision_type")
        statute["metadata"].pop("provision_number")
        results, _trace = legal_rag.retrieve_legal_context(
            "What did Kesavananda Bharati decide?",
            FakeIndex([statute, judgment]), FakeEmbedder(), top_k=2,
        )
        self.assertEqual(
            results[0]["metadata"]["source"],
            "Suprme-Court-Judgement_-Kesavananda-Bharati-v-State-of-Kerala-1973.pdf",
        )


def run_live_check() -> int:
    import os
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
    if not os.getenv("PINECONE_API_KEY"):
        print("Set PINECONE_API_KEY in backend/.env for live retrieval checks.")
        return 2
    from pinecone import Pinecone
    from sentence_transformers import SentenceTransformer

    client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = client.Index(os.getenv("PINECONE_INDEX", "legal-reference-m4"))
    model = SentenceTransformer(os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    ))
    cases = [
        (
            "According to Section 437(3), what conditions may a court impose?",
            "THE CODE OF CRIMINAL PROCEDURE, 1973.pdf", 180,
        ),
        (
            "What is the rule in Article 21?",
            "THE CONSTITUTION OF INDIA.pdf", 42,
        ),
        (
            "What does Article 14 of the Constitution guarantee?",
            "THE CONSTITUTION OF INDIA.pdf", 37,
        ),
        (
            "What does Article 19 of the Constitution protect?",
            "THE CONSTITUTION OF INDIA.pdf", 40,
        ),
        (
            "According to the Indian Penal Code, Section 302, what is the punishment for murder?",
            "THE INDIAN PENAL CODE.pdf", 74,
        ),
        (
            "What does Section 6 of the Right to Information Act 2005 provide about requests for information?",
            "THE RIGHT TO INFORMATION ACT, 2005.pdf", 7,
        ),
        (
            "What did Kesavananda Bharati decide?",
            "Suprme-Court-Judgement_-Kesavananda-Bharati-v-State-of-Kerala-1973.pdf", 1,
        ),
    ]
    failed = False
    for question, expected_source, expected_page in cases:
        passages, trace = legal_rag.retrieve_legal_context(
            question, index, model, os.getenv("PINECONE_NAMESPACE", ""), 6
        )
        top = passages[0]["metadata"] if passages else {}
        actual_source = top.get("source")
        actual_page = int(top["page"]) if top.get("page") is not None else None
        actual_pages = [
            int(page) for page in top.get("pages", [actual_page])
            if page is not None
        ]
        ok = (
            actual_source == expected_source
            and expected_page in actual_pages
        )
        print(
            f"{'PASS' if ok else 'FAIL'} {question}: "
            f"expected=({expected_source}, page {expected_page}) "
            f"actual=({actual_source}, pages {actual_pages})"
        )
        print(f"  trace={trace}")
        failed |= not ok
    out_of_scope = "What is the best way to bake a chocolate cake?"
    passages, trace = legal_rag.retrieve_legal_context(
        out_of_scope, index, model, os.getenv("PINECONE_NAMESPACE", ""), 6
    )
    if passages:
        print(
            f"FAIL {out_of_scope}: unexpected sources="
            f"{[(row['metadata'].get('source'), row['metadata'].get('page')) for row in passages]}"
        )
        failed = True
    else:
        from app import generate_answer
        answer, _answer_trace = generate_answer(out_of_scope)
        ok = "not found in the provided documents" in answer.lower()
        print(f"{'PASS' if ok else 'FAIL'} {out_of_scope}: {answer}")
        print(f"  trace={trace}")
        failed |= not ok
    return 1 if failed else 0


if __name__ == "__main__":
    if "--live" in sys.argv:
        raise SystemExit(run_live_check())
    unittest.main()
