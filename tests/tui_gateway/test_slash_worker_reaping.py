import queue


class _FakeProc:
    def __init__(self, return_code=0):
        self.return_code = return_code
        self.wait_called = False

    def wait(self):
        self.wait_called = True
        return self.return_code


def test_slash_worker_waiter_reaps_process_and_unblocks_queue():
    from tui_gateway.server import _SlashWorker

    worker = object.__new__(_SlashWorker)
    worker.proc = _FakeProc(return_code=0)
    worker.stdout_queue = queue.Queue()

    worker._wait_for_exit()

    assert worker.proc.wait_called is True
    assert worker.stdout_queue.get_nowait() is None
