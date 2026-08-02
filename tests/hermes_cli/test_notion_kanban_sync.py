from __future__ import annotations

from pathlib import Path

import json
import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import notion_kanban_sync as sync


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


class FakeNotion(sync.NotionClient):
    def __init__(self, pages):
        self.database_id = "fake-db"
        self.pages = pages
        self.updated = []
        self.created_pages = []
        self.activity = []

    def retrieve_database(self):
        return {
            "properties": {
                "Task": {"type": "title"},
                "Status": {"type": "select", "select": {"options": [
                    {"name": "Triage"}, {"name": "Todo"}, {"name": "Ready"},
                    {"name": "Running"}, {"name": "Blocked"}, {"name": "Done"}, {"name": "Archived"},
                ]}},
                "Notes": {"type": "rich_text"},
                "Source": {"type": "rich_text"},
                "Hermes Task ID": {"type": "rich_text"},
                "Last Synced At": {"type": "date"},
                "Sync Source": {"type": "rich_text"},
                "Sync Error": {"type": "rich_text"},
            }
        }

    def ensure_properties(self, *, dry_run, prune_status_options=False):
        return {"missing_statuses": [], "extra_statuses": [], "properties_added": [], "retired_properties": []}

    def query_tasks(self, *, page_size=100, since=None, limit=None):
        return self.pages[:limit] if limit else list(self.pages)

    def update_page_properties(self, page_id, properties):
        self.updated.append((page_id, properties))

    def create_page_for_hermes_task(self, task, *, schema):
        page_id = f"created-{task.id}"
        page = notion_page(page_id, task.title, sync.hermes_status_to_notion(task.status), hermes_task_id=task.id)
        page["properties"]["Source"] = {"type": "rich_text", "rich_text": [{"plain_text": f"Hermes Kanban sync: {task.id}"}]}
        self.created_pages.append(page)
        self.pages.append(page)
        return page

    def append_activity(self, page_id, text):
        self.activity.append((page_id, text))


def notion_page(
    page_id,
    title,
    status,
    hermes_task_id=None,
    last_edited_time="2026-01-01T00:10:00Z",
    assigned_agent="Dev",
):
    props = {
        "Task": {"type": "title", "title": [{"plain_text": title}]},
        "Status": {"type": "select", "select": {"name": status}},
        "Assigned Agent": {"type": "select", "select": {"name": assigned_agent}} if assigned_agent is not None else {"type": "select", "select": None},
        "Priority": {"type": "select", "select": {"name": "Medium"}},
        "Notes": {"type": "rich_text", "rich_text": []},
        "Source": {"type": "rich_text", "rich_text": []},
        "Hermes Task ID": {"type": "rich_text", "rich_text": []},
    }
    if hermes_task_id:
        props["Hermes Task ID"] = {"type": "rich_text", "rich_text": [{"plain_text": hermes_task_id}]}
    return {
        "id": page_id,
        "url": f"https://notion.test/{page_id}",
        "last_edited_time": last_edited_time,
        "archived": False,
        "properties": props,
    }


def test_new_notion_running_imports_as_ready_not_ghost_running(kanban_home, tmp_path):
    notion = FakeNotion([notion_page("page-running", "import me", "Running")])
    engine = sync.NotionKanbanSync(notion=notion, report_dir=tmp_path / "reports", state_path=tmp_path / "state.json")

    stats, _ = engine.run_once(dry_run=False, max_creates=5, sync_mode="two_way")

    assert stats.hermes_tasks_created == 1
    with kb.connect() as conn:
        tasks = kb.list_tasks(conn, include_archived=True)
        assert len(tasks) == 1
        task = tasks[0]
        assert task.status == "ready"
        assert task.claim_lock is None
        assert task.worker_pid is None
        assert task.current_run_id is None
    assert notion.updated
    props = notion.updated[-1][1]
    assert props["Status"]["select"]["name"] == "Ready"
    sync_error = props["Sync Error"]["rich_text"][0]["text"]["content"]
    assert "Only the dispatcher may set a task to running" in sync_error


def test_existing_task_ignores_notion_running_transition(kanban_home, tmp_path):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="paired", assignee="dev")
    notion = FakeNotion([notion_page("page-paired", "paired", "Running", hermes_task_id=task_id)])
    engine = sync.NotionKanbanSync(notion=notion, report_dir=tmp_path / "reports", state_path=tmp_path / "state.json")

    stats, _ = engine.run_once(dry_run=False, sync_mode="two_way")

    assert stats.conflicts == 1
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task.status == "ready"
        comments = kb.list_comments(conn, task_id)
        assert any("dispatcher owns runtime state" in c.body for c in comments)
    props = notion.updated[-1][1]
    assert props["Status"]["select"]["name"] == "Ready"
    sync_error = props["Sync Error"]["rich_text"][0]["text"]["content"]
    assert "kept Hermes status 'ready'" in sync_error


def test_new_notion_task_with_invalid_assigned_agent_imports_to_triage_without_assignee(kanban_home, tmp_path):
    notion = FakeNotion([
        notion_page("page-invalid-agent", "needs routing", "Ready", assigned_agent="Orion CC")
    ])
    engine = sync.NotionKanbanSync(
        notion=notion,
        report_dir=tmp_path / "reports",
        state_path=tmp_path / "state.json",
    )

    stats, _ = engine.run_once(dry_run=False, max_creates=5, sync_mode="two_way")

    assert stats.hermes_tasks_created == 1
    assert stats.conflicts == 1
    with kb.connect() as conn:
        tasks = kb.list_tasks(conn, include_archived=True)
        assert len(tasks) == 1
        task = tasks[0]
        assert task.status == "triage"
        assert task.assignee is None
        assert task.claim_lock is None
        assert task.worker_pid is None
        assert task.current_run_id is None
    props = notion.updated[-1][1]
    assert props["Status"]["select"]["name"] == "Triage"
    sync_error = props["Sync Error"]["rich_text"][0]["text"]["content"]
    assert "Invalid Notion Assigned Agent 'Orion CC'" in sync_error
    assert "Imported into Hermes triage" in sync_error


def test_existing_task_ignores_invalid_notion_assigned_agent(kanban_home, tmp_path):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="paired", assignee="dev")
    notion = FakeNotion([
        notion_page(
            "page-invalid-agent-existing",
            "paired",
            "Ready",
            hermes_task_id=task_id,
            assigned_agent="Terminal Lane",
        )
    ])
    engine = sync.NotionKanbanSync(
        notion=notion,
        report_dir=tmp_path / "reports",
        state_path=tmp_path / "state.json",
    )

    stats, _ = engine.run_once(dry_run=False, sync_mode="two_way")

    assert stats.conflicts == 1
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task.assignee == "dev"
        assert task.status == "ready"
        comments = kb.list_comments(conn, task_id)
        assert any("Invalid Notion Assigned Agent 'Terminal Lane'" in c.body for c in comments)
    props = notion.updated[-1][1]
    sync_error = props["Sync Error"]["rich_text"][0]["text"]["content"]
    assert "kept Hermes assignee 'dev'" in sync_error


def test_default_mode_is_outbound_only_and_refuses_notion_import(kanban_home, tmp_path):
    notion = FakeNotion([notion_page("page-new", "do not import", "Ready")])
    engine = sync.NotionKanbanSync(notion=notion, report_dir=tmp_path / "reports", state_path=tmp_path / "state.json")

    stats, report_path = engine.run_once(dry_run=False)

    assert stats.hermes_tasks_created == 0
    assert stats.notion_to_hermes_updates == 0
    with kb.connect() as conn:
        assert kb.list_tasks(conn, include_archived=True) == []
    payload = json.loads(report_path.read_text())
    assert payload["sync_mode"] == "outbound_only"
    assert payload["outbound_only"] is True


def test_outbound_only_dry_run_reports_no_inbound_mutation(kanban_home, tmp_path):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="dry-run mirror", assignee="dev")
    notion = FakeNotion([notion_page("page-new", "do not import", "Ready")])
    engine = sync.NotionKanbanSync(notion=notion, report_dir=tmp_path / "reports", state_path=tmp_path / "state.json")

    stats, report_path = engine.run_once(dry_run=True, sync_mode="outbound_only", hermes_task_ids={task_id})

    assert stats.hermes_tasks_created == 0
    assert stats.notion_to_hermes_updates == 0
    assert stats.notion_pages_would_create == 1
    assert stats.hermes_to_notion_would_update == 1
    payload = json.loads(report_path.read_text())
    assert payload["sync_mode"] == "outbound_only"
    assert payload["outbound_only"] is True
    assert payload["stats"]["hermes_tasks_created"] == 0
    assert payload["stats"]["notion_to_hermes_updates"] == 0


def test_outbound_only_still_creates_notion_pages_for_hermes_tasks(kanban_home, tmp_path):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="mirror me", assignee="dev")
    notion = FakeNotion([])
    engine = sync.NotionKanbanSync(notion=notion, report_dir=tmp_path / "reports", state_path=tmp_path / "state.json")

    stats, _ = engine.run_once(dry_run=False, sync_mode="outbound_only", hermes_task_ids={task_id})

    assert stats.hermes_tasks_created == 0
    assert stats.notion_to_hermes_updates == 0
    assert stats.notion_pages_created == 1
    assert stats.hermes_to_notion_updates == 1
    assert notion.created_pages[0]["properties"]["Hermes Task ID"]["rich_text"][0]["plain_text"] == task_id


def test_configured_sync_mode_defaults_outbound_and_requires_explicit_import(monkeypatch):
    monkeypatch.delenv("HERMES_NOTION_KANBAN_SYNC_MODE", raising=False)
    monkeypatch.delenv("HERMES_NOTION_KANBAN_ALLOW_INBOUND", raising=False)
    monkeypatch.setattr(sync, "load_config", lambda: {})

    assert sync.configured_sync_mode() == "outbound_only"
    assert sync.configured_sync_mode(enable_notion_import=True) == "two_way"

    monkeypatch.setattr(sync, "load_config", lambda: {"notion_kanban_sync": {"mode": "two_way"}})
    assert sync.configured_sync_mode() == "two_way"
    assert sync.configured_sync_mode(outbound_only=True) == "outbound_only"
