from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from chatkit.agents import AgentContext
from chatkit.types import CustomTask, ThreadMetadata

from multimedia_intelligence.chat.tool_activity import ToolActivityReporter
from multimedia_intelligence.context import ClientInfo, RequestContext


class ActivityStore:
    def generate_item_id(self, item_type: str, thread: object, context: object) -> str:
        del thread, context
        return f"{item_type}_activity"


async def test_server_tool_activity_starts_completes_and_streams_a_card() -> None:
    agent_context = AgentContext(
        thread=ThreadMetadata(id="thread_1", created_at=datetime.now(UTC)),
        store=ActivityStore(),  # type: ignore[arg-type]
        request_context=RequestContext(client=ClientInfo(user_id="user_1", username="user")),
    )
    hook_context = SimpleNamespace(
        context=agent_context,
        tool_arguments='{"query":"roadmap"}',
    )
    reporter = ToolActivityReporter("thread_1")

    await reporter.start(hook_context, "file_search", "call_1")
    await reporter.end(
        hook_context,
        "file_search",
        "call_1",
        {"query": "roadmap", "collection": {"name": "Docs"}, "results": []},
    )

    assert agent_context.workflow_item is not None
    task = agent_context.workflow_item.workflow.tasks[0]
    assert isinstance(task, CustomTask)
    assert task.status_indicator == "complete"
    assert task.content == "Found 0 collection matches"
    events: list[Any] = []
    while not agent_context._events.empty():
        events.append(agent_context._events.get_nowait())
    assert [event.type for event in events] == [
        "thread.item.added",
        "thread.item.updated",
        "thread.item.done",
    ]


async def test_browser_tool_activity_remains_loading_until_continuation() -> None:
    agent_context = AgentContext(
        thread=ThreadMetadata(id="thread_1", created_at=datetime.now(UTC)),
        store=ActivityStore(),  # type: ignore[arg-type]
        request_context=RequestContext(client=ClientInfo(user_id="user_1", username="user")),
    )
    hook_context = SimpleNamespace(context=agent_context, tool_arguments='{"page":1}')
    reporter = ToolActivityReporter("thread_1")

    await reporter.start(hook_context, "list_files", "call_1")
    await reporter.end(
        hook_context,
        "list_files",
        "call_1",
        {"client_tool": "list_files", "status": "waiting_for_browser"},
    )

    assert agent_context.workflow_item is not None
    task = agent_context.workflow_item.workflow.tasks[0]
    assert isinstance(task, CustomTask)
    assert task.status_indicator == "loading"
    assert task.content == "Waiting for the browser workspace result…"
