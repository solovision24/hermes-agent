"""Windows deliberately refuses durable webhook storage without a verified private ACL."""

import pytest

from agent.outbound_webhook_spool import Spool
from agent.outbound_webhooks import WebhookTarget, _build_delivery, _serialize_payload


@pytest.mark.platforms("windows")
def test_opt_in_fails_closed_instead_of_inheriting_permissive_acl(tmp_path):
    target = WebhookTarget(url="https://example.test/hook", events=["on_session_end"], durable=True)
    identity = "a" * 32
    body = _serialize_payload("on_session_end", {"session_id": "test"}, identity)
    assert not Spool(tmp_path).enqueue(target, _build_delivery("on_session_end", target, body, identity))
    assert not (tmp_path / "outbound-webhooks" / "spool.db").exists()
