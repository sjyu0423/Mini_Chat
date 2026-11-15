# M6.3 pre-registration addendum (2026-11-15, before attempt 2; user-approved)

The pre-registration (`preregistration.md`, SHA-256 `563dd005…`) is unchanged. This addendum records
one deviation in the *fixed training recipe*, made after attempt 1 aborted and before any M6.3
output was generated or inspected:

* **Attempt 1** (`runs/m6_3_sft_30m-20261114-234714_attempt1_guard_abort`) stopped at step
  2,224 / 3,153 when the MPS memory guard tripped (pool 11.0 GB > 10 GB) with
  `torch.mps.empty_cache()` every 25 steps (the M5 setting).
* **Attempt 2** releases the MPS cache **every step** (`empty_cache_interval = 1`). Fresh-process
  measurement, 800 real steps: peak pool 8.66 GB, 0.303 s/step (interval 5: 8.74 GB, 0.290 s/step).
  This is a memory-management setting only: data, order, batches, loss, optimizer, schedule and
  seed are identical to attempt 1 and to M5. The 10 GB guard is unchanged.
* Evaluation and promotion rules are unchanged. Attempt 1's checkpoint is not evaluated.
