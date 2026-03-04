# mlserverpy

git submodule add -b mlserverpy git@github.com:MathiasKES/ML-Server.git


Small Python SDK for ML-Server.

## Quickstart

```python
import mlserverpy as ml

ml.init(
    host="http://127.0.0.1:5001",
    username="admin",
    password="password",
    flush_interval=2.0,
    offline_mode="queue",
)

run_id = ml.run(
    name="test-run",
    dataset="cifar100",
    methods=["DLG", "iDLG"],
    iterations=300,
    lr=0.5,
    num_classes=100,
)

for step in range(100):
    ml.log_metric(method="DLG", metric="loss", value=10.0/(step+1), step=step)
    ml.log_metric(method="DLG", metric="mse", value=1.0/(step+1), step=step)

ml.post(kind="log", text="Training finished")
ml.flush()
```

## Notes
* Metrics are buffered locally and sent periodically.

* If the server is offline, events are queued on disk and can be replayed with ml.sync().

