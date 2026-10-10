"""Offline Flask API checks using an in-memory SQLite database."""
from __future__ import annotations

import base64
import os
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

os.environ["DATABASE_URL"] = "sqlite://"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import app as backend


class FlaskApiTests(unittest.TestCase):
    def setUp(self):
        self.context = backend.app.app_context()
        self.context.push()
        backend.db.drop_all()
        backend.db.create_all()
        backend.app.config["TESTING"] = True
        self.client = backend.app.test_client()

    def tearDown(self):
        backend.db.session.remove()
        backend.db.drop_all()
        self.context.pop()

    def signup(self, client=None):
        client = client or self.client
        response = client.post("/api/auth/signup", json={
            "name": "Test Researcher",
            "email": "test-" + uuid.uuid4().hex + "@example.invalid",
            "password": "a-safe-test-password",
        })
        self.assertEqual(response.status_code, 201, response.get_json())
        return response.get_json()["user"]

    def test_signup_me_and_logout(self):
        user = self.signup()
        self.assertEqual(self.client.get("/api/auth/me").get_json()["user"]["id"], user["id"])
        self.assertEqual(self.client.post("/api/auth/logout").status_code, 200)
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)

    def test_authentication_is_required_for_conversations(self):
        self.assertEqual(self.client.get("/api/conversations").status_code, 401)
        self.signup()
        self.assertEqual(self.client.get("/api/conversations").status_code, 200)

    def test_conversation_is_private_and_messages_are_saved(self):
        self.signup()
        created = self.client.post("/api/conversations", json={})
        self.assertEqual(created.status_code, 201)
        conversation_id = created.get_json()["conversation"]["id"]
        with patch.object(backend, "generate_answer", return_value=("A source-grounded test answer.", [])):
            answer = self.client.post(
                f"/api/conversations/{conversation_id}/messages",
                json={"message": "What does Section 7 say?"},
            )
        self.assertEqual(answer.status_code, 200, answer.get_json())
        self.assertEqual(answer.get_json()["answer"], "A source-grounded test answer.")
        stored = self.client.get(f"/api/conversations/{conversation_id}").get_json()["conversation"]
        self.assertEqual([message["role"] for message in stored["messages"]], ["user", "assistant"])

        other_client = backend.app.test_client()
        self.signup(other_client)
        self.assertEqual(other_client.get(f"/api/conversations/{conversation_id}").status_code, 404)

    def test_profile_name_and_photo_can_be_updated_and_removed(self):
        user = self.signup()
        photo_bytes = b"\x89PNG\r\n\x1a\nprofile-test"
        photo = "data:image/png;base64," + base64.b64encode(photo_bytes).decode("ascii")
        updated = self.client.put("/api/auth/profile", json={
            "name": "Updated Researcher",
            "photo": photo,
        })
        self.assertEqual(updated.status_code, 200, updated.get_json())
        self.assertEqual(updated.get_json()["user"]["name"], "Updated Researcher")
        self.assertEqual(updated.get_json()["user"]["photo"], photo)
        self.assertEqual(self.client.get("/api/auth/me").get_json()["user"]["photo"], photo)

        renamed = self.client.put("/api/auth/profile", json={"name": "Another Name"})
        self.assertEqual(renamed.status_code, 200)
        self.assertEqual(renamed.get_json()["user"]["photo"], photo)
        removed = self.client.put("/api/auth/profile", json={
            "name": "Another Name",
            "remove_photo": True,
        })
        self.assertEqual(removed.status_code, 200)
        self.assertIsNone(removed.get_json()["user"]["photo"])
        self.assertEqual(backend.db.session.get(backend.User, user["id"]).profile, None)

    def test_profile_rejects_invalid_photo_without_changing_account(self):
        user = self.signup()
        response = self.client.put("/api/auth/profile", json={
            "name": "Must Not Be Saved",
            "photo": "data:image/png;base64," + base64.b64encode(b"not a PNG").decode("ascii"),
        })
        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            self.client.get("/api/auth/me").get_json()["user"]["name"],
            user["name"],
        )

    def test_chat_can_be_deleted_with_its_messages(self):
        self.signup()
        conversation_id = self.client.post("/api/conversations", json={}).get_json()[
            "conversation"
        ]["id"]
        with patch.object(backend, "generate_answer", return_value=("Answer.", [])):
            self.client.post(
                f"/api/conversations/{conversation_id}/messages",
                json={"message": "A question"},
            )
        self.assertEqual(backend.Message.query.count(), 2)
        deleted = self.client.delete(f"/api/conversations/{conversation_id}")
        self.assertEqual(deleted.status_code, 200)
        self.assertTrue(deleted.get_json()["ok"])
        self.assertEqual(backend.Message.query.count(), 0)
        self.assertEqual(self.client.get(f"/api/conversations/{conversation_id}").status_code, 404)

    def test_chat_deletion_is_private_and_requires_authentication(self):
        self.signup()
        conversation_id = self.client.post("/api/conversations", json={}).get_json()[
            "conversation"
        ]["id"]
        other_client = backend.app.test_client()
        self.assertEqual(other_client.delete(
            f"/api/conversations/{conversation_id}"
        ).status_code, 401)
        self.signup(other_client)
        self.assertEqual(other_client.delete(
            f"/api/conversations/{conversation_id}"
        ).status_code, 404)
        self.assertEqual(self.client.get(f"/api/conversations/{conversation_id}").status_code, 200)

    def test_unretrievable_question_is_reported_as_not_found(self):
        with patch.object(backend, "initialize_rag", return_value=(object(), object())), \
                patch.object(backend, "retrieve_legal_context", return_value=([], [])):
            answer, trace = backend.generate_answer("What is the best way to bake a cake?")
        self.assertIn("not found in the provided documents", answer.lower())
        self.assertEqual(trace, [])

    def test_ambiguous_section_prompts_for_act(self):
        trace = [{
            "mode": "ambiguous-provision",
            "provision": "Section 437",
            "sources": ["The Companies Act.pdf", "The Code of Criminal Procedure.pdf"],
        }]
        with patch.object(backend, "initialize_rag", return_value=(object(), object())), \
                patch.object(backend, "retrieve_legal_context", return_value=([], trace)):
            answer, returned_trace = backend.generate_answer("What is Section 437?")
        self.assertIn("Please specify the Act or document", answer)
        self.assertIn("The Code of Criminal Procedure.pdf", answer)
        self.assertEqual(returned_trace, trace)

    def test_generated_citations_are_replaced_with_retrieval_pages(self):
        answer = (
            "The court shall impose the stated conditions.\n\n"
            "Citations & sources\n- Multi-page Act.pdf, page 999"
        )
        passages = [{
            "metadata": {
                "source": "Multi-page Act.pdf",
                "page": 180,
                "pages": [180, 181],
            },
        }]
        result = backend._with_verified_citations(answer, passages)
        self.assertIn("pages 180, 181", result)
        self.assertNotIn("page 999", result)
        self.assertIn("shall impose", result)

    def test_answer_prompt_preserves_modal_verbs_and_source_pages(self):
        messages = backend._ollama_messages(
            "What conditions apply?",
            "[1] Source: CrPC.pdf, pages 180, 181\nThe Court shall impose conditions.",
            [],
        )
        prompt = messages[0]["content"]
        self.assertIn("'shall' as mandatory and 'may' as discretionary", prompt)
        self.assertIn("exact source names and page numbers", prompt)
        self.assertIn("context is incomplete", prompt)

    def test_downstream_answer_failure_returns_visible_service_error(self):
        self.signup()
        conversation_id = self.client.post("/api/conversations", json={}).get_json()[
            "conversation"
        ]["id"]
        for failure in ("Pinecone unavailable", "Ollama unavailable"):
            with patch.object(
                backend, "generate_answer", side_effect=RuntimeError(failure)
            ):
                response = self.client.post(
                    f"/api/conversations/{conversation_id}/messages",
                    json={"message": "Question during service failure."},
                )
            self.assertEqual(response.status_code, 503)
            self.assertIn("Check Ollama, Pinecone", response.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
