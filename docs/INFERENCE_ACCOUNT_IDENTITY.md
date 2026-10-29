# Account identity at inference

Requests contain `payload.sourceAccountId` and `payload.destinationAccountId`.
These are opaque identities, not learned numeric features. Mongo ObjectID strings
are valid; converting them to synthetic training UUIDs is unnecessary.

`ReplayPredictor.account_identity_policy = "distinct-account-nodes-v1"` advertises
the runtime policy used by the Anomaly adapter. Older predictors lack this marker;
the adapter must keep rejecting unsupported identities with those predictors.

For validated TransferCreated/TransferCompleted events in the strict prior prefix,
and for the current transfer request, the predictor builds a per-request catalog:

- Preserve frozen bundle node IDs, types, creation times and indices.
- Append each new account once, sorted by identity, as a separate account node.
- Never merge different real accounts into `__unknown__` or training accounts.
- New nodes have availability at the earlier of first observation and the graph
  cutoff. This is inference availability, not a claim about an account opening date.
  A new target can therefore retain its identity as a cold-start node even when
  the graph snapshot precedes its first observed request.
- Keep the existing graph snapshot cutoff, strict prior history, feature schema,
  train-fitted scalers, weights and threshold. Only current account identity/type
  can come from the target; no current/future settlement enters graph/history.
- Build the catalog from each request's validated prefix, without mutating the
  bundle or keeping cross-request state. Restarts and retries reconstruct it.

This support is limited to transfer account endpoints. Counterparty typing for
other transaction types retains the existing bundle-catalog behavior.

Existing catalog inputs retain offline tensor/score parity. Tests also cover
unseen ObjectIDs, empty history, same-snapshot cold starts, renamed identity/graph
parity, future-event exclusion, and restart stability. This validates compatibility,
not fraud detection accuracy or calibration on real customer data. No production
training is run and no bundle artifact is rewritten by this change.
