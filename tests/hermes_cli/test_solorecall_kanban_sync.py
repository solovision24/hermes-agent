from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db
from hermes_cli import solorecall_kanban_sync as sync


class FakeSoLoRecall:
    def __init__(self, blocks=None, *, fail_list=False):
        self.blocks = list(blocks or [])
        self.created = []
        self.updated = []
        self.archived = []
        self.fail_list = fail_list

    def list_blocks(self, *, limit=100, offset=0, block_type=None):
        if self.fail_list:
            raise sync.SoLoRecallError("boom")
        return self.blocks[:limit]

    def create_block(self, payload):
        block_id = f"block-{len(self.created) + 1}"
        block = {"id": block_id, **payload, "updated_at": "2026-01-01T00:00:00Z"}
        self.created.append(payload)
        self.blocks.append(block)
        return block

    def update_block(self, block_id, payload):
        self.updated.append((block_id, payload))
        for block in self.blocks:
            if block.get("id") == block_id:
                block.update(payload)
                block.setdefault("properties", {}).update(payload.get("properties", {}))
                return block
        return {"id": block_id, **payload}

    def archive_block(self, block_id):
        self.archived.append(block_id)
        return {"id": block_id, "archived": True}


def block(block_id, title, status="Ready", hermes_task_id=None, archived=False, in_trash=False, priority=0):
    props = {"title": title, "status": status, "priority": priority}
    if hermes_task_id:
        props["hermes_task_id"] = hermes_task_id
        props[sync.MARKER_KEY] = {"task_id": hermes_task_id, "version": sync.MARKER_VERSION}
    return {
        "id": block_id,
        "type": "page",
        "properties": props,
        "content": {"body": f"body for {title}"},
        "archived": archived,
        "in_trash": in_trash,
        "updated_at": "2026-01-01T00:00:00Z",
    }


@pytest.fixture()
def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "hermes-home"))
    kanban_db.init_db()

    def make(fake):
        return sync.SoLoRecallKanbanSync(
            solorecall=fake,
            report_dir=tmp_path / "reports",
            state_path=tmp_path / "mapping.json",
        )

    return make


def create_task(title="task", *, status=None, priority=0):
    conn = kanban_db.connect()
    try:
        task_id = kanban_db.create_task(conn, title=title, body="body", created_by="tester", priority=priority)
        if status and status != "ready":
            with kanban_db.write_txn(conn):
                conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
        return task_id
    finally:
        conn.close()


def get_task(task_id):
    conn = kanban_db.connect()
    try:
        return kanban_db.get_task(conn, task_id)
    finally:
        conn.close()


def test_dry_run_reports_outbound_create_without_mutation(engine):
    create_task("dry run task")
    fake = FakeSoLoRecall([])

    result = engine(fake).run_once(apply=False, limit=25)

    assert result["stats"]["solorecall_blocks_would_create"] == 1
    assert fake.created == []
    assert Path(result["report"]).exists()


def test_apply_outbound_create_and_mapping(engine):
    task_id = create_task("create me", priority=7)
    fake = FakeSoLoRecall([])

    result = engine(fake).run_once(apply=True, limit=25)

    assert result["stats"]["solorecall_blocks_created"] == 1
    assert fake.created[0]["properties"]["hermes_task_id"] == task_id
    mapping = sync.MappingStore(Path(result["state"]))
    assert mapping.block_for_task(task_id) == "block-1"


def test_apply_outbound_update_existing_block(engine):
    task_id = create_task("new title", status="blocked", priority=9)
    fake = FakeSoLoRecall([block("sr-1", "old title", "Ready", hermes_task_id=task_id, priority=1)])

    result = engine(fake).run_once(apply=True, limit=25)

    assert result["stats"]["solorecall_blocks_updated"] == 1
    assert fake.updated[0][0] == "sr-1"
    assert fake.updated[0][1]["properties"]["status"] == "Blocked"
    assert fake.updated[0][1]["properties"]["title"] == "new title"


def test_two_way_imports_unmapped_solorecall_block(engine):
    fake = FakeSoLoRecall([block("sr-new", "import me", "Todo")])

    result = engine(fake).run_once(apply=True, limit=25, mode="two_way")

    assert result["stats"]["hermes_tasks_created"] == 1
    tasks = kanban_db.list_tasks(kanban_db.connect())
    imported = [t for t in tasks if t.title == "import me"]
    assert imported
    assert imported[0].status == "todo"
    assert fake.updated[0][0] == "sr-new"


def test_two_way_updates_existing_hermes_task(engine):
    task_id = create_task("old", status="ready", priority=0)
    fake = FakeSoLoRecall([block("sr-1", "renamed", "Blocked", hermes_task_id=task_id, priority=11)])

    result = engine(fake).run_once(apply=True, limit=25, mode="two_way")

    task = get_task(task_id)
    assert result["stats"]["hermes_tasks_updated"] == 1
    assert task.title == "renamed"
    assert task.status == "blocked"
    assert task.priority == 11


def test_outbound_only_skips_unmapped_inbound_block(engine):
    fake = FakeSoLoRecall([block("sr-new", "do not import", "Ready")])

    result = engine(fake).run_once(apply=True, limit=25, mode="outbound_only")

    assert result["stats"]["inbound_skipped_outbound_only"] == 1
    tasks = kanban_db.list_tasks(kanban_db.connect())
    assert all(t.title != "do not import" for t in tasks)


def test_api_error_is_reported(engine):
    create_task("api error")
    fake = FakeSoLoRecall(fail_list=True)

    with pytest.raises(sync.SoLoRecallError):
        engine(fake).run_once(apply=False, limit=25)


def test_no_hard_delete_for_archived_or_trashed_blocks(engine):
    task_id = create_task("keep me", status="ready")
    fake = FakeSoLoRecall([block("sr-archived", "remote archived", "Archived", hermes_task_id=task_id, archived=True)])

    result = engine(fake).run_once(apply=True, limit=25, mode="two_way")

    assert result["stats"]["no_hard_delete_skips"] >= 1
    assert get_task(task_id).status == "ready"
    assert fake.archived == []
