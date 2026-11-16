# Inbound admission plugins

`kind: hook` plugins may call `ctx.inbound(handler)` during setup. Harness stores
registrations and owns cleanup; it does not import Gateway or process IM events.
The host must explicitly connect these handlers to its admission API.

```python
from octop_harness.plugins import InboundDispatch

class Handler:
    async def handle(self, channel, message, dispatch: InboundDispatch) -> bool:
        # False leaves this input with the host's normal routing.
        return False

    def close(self) -> None:
        # Cancel tasks and timers synchronously. Do not start more work.
        pass

def setup(ctx):
    ctx.inbound(Handler())
```

Return `True` only when the plugin owns the input. A delayed task can call
`await dispatch(message, activate=True, receipt=receipt)` when supported by the
host. Gateway retains the session lock and rechecks channel registration and
group permissions; this callback does not grant permission to bypass visibility,
sender authorization or tool limits. Do not retain callbacks after `close()`.

Replacing, clearing or unregistering a plugin invokes its handlers' `close()`.
Cleanup failures are logged and do not prevent other handlers from closing.
`async def close()` is rejected: hosts that need asynchronous draining may call
an optional `wait_closed()` after synchronous cancellation. Cleanup should be
idempotent. Failed plugin updates must reload a fresh instance of a closed hook.

## Optional settings and text rendering

Any plugin kind can register one `ctx.settings(provider)`. The provider implements
`async get_settings(context) -> dict` and `async save_settings(context, values)`.
It owns its schema, validation, defaults, inheritance and storage. A host decides
which common, credential-free catalog data to include in `context` and controls
authentication and authorization. Harness does not interpret business fields or
provide an HTTP server. `PluginSettingsProvider` is exported for type checking.

An inbound handler can also implement `InboundTextRenderer.render_inbound_text(
message, current_text, *, background, locale) -> str | None`. A supporting host
invokes it only while assembling model input, after command routing and media
materialization. `current_text` includes prepared attachment/document hints;
`background` contains dictionaries with `sender_id`, `sender_name`, and text
`segments`. Return `None` to retain host rendering, or text to replace the user
turn's text block. Separate image blocks, routing and permissions stay with the
host. Plugins should claim only their own messages and never parse or replace
hardcoded host prompt prefixes. Hosts without this optional extension continue
their original rendering.

## Optional decision scoring

A handler may implement `DecisionScorerAware.bind_decision_scorer(provider)`.
The host passes a `DecisionScorerProvider`: an async callable returning a scorer
or `None`. The scorer accepts `state`, `statements` and optional `strict`, and
returns probabilities keyed by statement ID. Providers should read current host
configuration so disabling or changing the service affects later calls.

This capability is optional. Plugins must handle an absent provider, `None`,
timeouts and scoring failures using their own bounded fallback. Scoring can send
message content to the configured remote service; plugin authors must disclose
that behavior and control the amount of context sent. Credentials stay in the
host's client and must not be copied into plugin configuration or UI responses.
