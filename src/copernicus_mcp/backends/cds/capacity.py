"""Pure capacity-vs-content classifier for empty-log remote failures
(T-CDS-RESIL-001).

CDS / ADS / EWDS report a failure with no server-side log both for a malformed
request and for a refusal under load — and the two need opposite responses
(change the request vs resubmit it unchanged after a pause). The flattened
error text cannot tell them apart, so the classifier corroborates:

- the job's own remote status is ``rejected`` — the store's **admission
  control** turned it away before it ever ran. That is a refusal, never a
  verdict on the data: the recorded CORDEX incident came back
  ``status: "rejected"``, ``log: []``, and the server-side reason was "Number
  queued requests for this dataset is temporarily limited"; the identical
  requests succeeded when resubmitted after the burst drained; or
- a **sibling chunk** of the same chunked parent already succeeded — the
  request shape is proven good, so the refusal was about timing, not content.

A job that reached ``failed`` on its own, with no log and no successful
sibling, stays ``unknown`` and is never retried. That is the shape of a
request that RAN and found nothing — a version/period combination that is
valid by membership but serves no data fails this way within seconds, and
resubmitting it is only a slower way to fail.

Deliberately NOT used: a census of how many jobs the account has in flight.
Three review rounds showed it cannot discriminate. Our own submissions are
paced per retrieval, so the total rises simply because we are busy; and the
one refusal on record was caused by our own burst against a **per-dataset**
queue cap with nothing else on the account, so neither the total nor the
foreign remainder identifies it. The remote status does, directly.
"""

from __future__ import annotations

from typing import Literal

FailureClassification = Literal["capacity_suspected", "content", "unknown"]


def classify_remote_failure(
    *,
    empty_log: bool,
    sibling_succeeded: bool,
    remote_status_rejected: bool,
) -> FailureClassification:
    """Classify a terminal remote-job failure.

    ``empty_log`` — the server returned no failure message in any known slot.
    ``sibling_succeeded`` — another chunk of the same parent is ``successful``.
    ``remote_status_rejected`` — the store reported this job's status as
    ``rejected`` (refused at admission rather than run and failed).
    """
    if not empty_log:
        return "content"
    if remote_status_rejected or sibling_succeeded:
        return "capacity_suspected"
    return "unknown"
