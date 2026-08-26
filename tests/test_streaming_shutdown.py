from __future__ import annotations

import threading


def test_stop_equity_feed_joins_partial_staleness_thread_without_client(monkeypatch):
    import trader_koo.streaming.service as service

    monkeypatch.setattr(service, "_client", None)
    monkeypatch.setattr(service, "_staleness_running", True)
    service._staleness_stop_event.clear()
    thread = threading.Thread(target=service._staleness_loop, name="partial-equity-monitor")
    monkeypatch.setattr(service, "_staleness_thread", thread)
    thread.start()
    service.stop_equity_feed()
    assert not thread.is_alive()
