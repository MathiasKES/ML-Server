# mlserverpy

```bash
git submodule add -b mlserverpy git@github.com:MathiasKES/ML-Server.git
```

Small Python SDK for ML-Server.

## Quickstart

```python
import mlserverpy

client = mlserverpy.Client(
    host="http://localhost:5001",
    username="admin",
    password="password",
    offline_mode="queue",
    flush_interval=2.0,
)

run_id = client.run(name="demo", dataset="cifar100", methods=["DLG","iDLG"], iterations=300)

for step in range(100):
    client.log_metric(method="DLG", metric="loss", value=10.0/(step+1), step=step)
    client.log_metric(method="DLG", metric="mse", value=1.0/(step+1), step=step)

client.flush()
```

## Offline mode

- `offline_mode="queue"`: store failed requests under `~/.cache/mlserverpy/spool/<run_id>` and replay with `client.sync()`
- `offline_mode="drop"`: silently drop failed requests
- `offline_mode="raise"`: raise on failures immediately


## Notes
* Metrics are buffered locally and sent periodically.

* If the server is offline, events are queued on disk and can be replayed with ml.sync().

