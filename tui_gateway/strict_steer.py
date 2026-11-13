"""Run-bound steering receipts. Never enters the legacy next-turn steer slot.

Receipts survive transport reconnects, not backend/session retirement. A retired run
identifier is never accepted again, so an ambiguous client must reconcile, not resend
under a fresh identifier. Delivery means insertion at a real tool checkpoint, not
that the provider has acted on the text.
"""
from __future__ import annotations

import logging
import threading
import uuid

from agent.prompt_builder import steer_user_row


class StrictSteerRun:
    MAX_INTENTS = 128

    def __init__(self):
        self.run_id = uuid.uuid4().hex
        self.lock = threading.RLock()
        self.closed = False
        self.receipts = {}
        self.emit = None

    def snapshot(self):
        with self.lock:
            return {"run_id": self.run_id, "active": not self.closed,
                    "receipts": [self._public(row) for row in self.receipts.values()]}

    def _public(self, row):
        return {"run_id": self.run_id, "intent_id": row["intent_id"],
                "status": row["status"]}

    def submit(self, intent_id, text):
        with self.lock:
            previous = self.receipts.get(intent_id)
            if previous is not None:
                if previous["text"] != text:
                    raise ValueError("intent_id already bound to different text")
                return self._public(previous)
            if self.closed:
                raise ValueError("run is terminal")
            if len(self.receipts) >= self.MAX_INTENTS:
                raise ValueError("run steering receipt capacity reached")
            row = {"intent_id": intent_id, "text": text, "status": "accepted"}
            self.receipts[intent_id] = row
            self._notify(row)
            return self._public(row)

    def _notify(self, row):
        if self.emit is not None:
            try:
                self.emit(self._public(row))
            except Exception:
                # Transport loss cannot roll back checkpoint insertion. The receipt
                # remains authoritative and can be polled after reattachment.
                logging.getLogger(__name__).warning("strict steer event unavailable", exc_info=True)

    def checkpoint(self, messages, num_tool_msgs, drain_legacy):
        # A previous turn's tool row is NOT a checkpoint. Only the just-completed
        # tool batch invokes this method, and its newest row must still be tool.
        if num_tool_msgs <= 0 or not messages or messages[-1].get("role") != "tool":
            return False
        with self.lock:
            if self.closed:
                return False
            pending = [r for r in self.receipts.values() if r["status"] == "accepted"]
            texts = [r["text"] for r in pending]
            legacy = drain_legacy()
            if legacy:
                texts.append(legacy)
            if texts:
                messages.append(steer_user_row("\n".join(texts)))
            for row in pending:
                row["status"] = "delivered"
                self._notify(row)
            return True

    def finish(self):
        with self.lock:
            self.closed = True
            for row in self.receipts.values():
                if row["status"] == "accepted":
                    row["status"] = "missed"
                    self._notify(row)


def start_run(session):
    previous = session.get("_strict_steer_run")
    if previous is not None:
        previous.finish()
    session["_strict_steer_run"] = StrictSteerRun()


def finish_run(session):
    run = session.get("_strict_steer_run")
    if run is not None:
        run.finish()
