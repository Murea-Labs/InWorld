# nWorld

**English** | [中文](README_zh.md)

> **Status:** the Murea-Fast model weights are not published yet. They will be released on
> Hugging Face, and the link will be added to the [Murea-Fast README](inference/README.md#model).
> Until then the inference service cannot be started, so the agent has no video model to
> film with.

nWorld is an open-source video generation system. It generates videos with synchronized
audio from text prompts and ordered reference images, continues existing videos, and
includes an agent that turns a player's actions into a continuing, filmed story.

| Component | Directory | What it is |
| --- | --- | --- |
| **Murea-Fast** | [`inference/`](inference/README.md) | The inference service: serves the model over an HTTP API |
| **Murea Agent** | [`agent/`](agent/README.md) | An open-world video agent with a web UI: plans each beat with a language model and films it on Murea-Fast |

## Getting started

1. Start a Murea-Fast service on a GPU machine: see the [Murea-Fast README](inference/README.md).
2. Point Murea Agent at it with `MUREA_URL` and run it: see the
   [Murea Agent README](agent/README.md).
3. Optionally, switch the agent to co-op rooms for 2–4 players: see
   [Co-op mode](agent/README.md#co-op-mode).

The [API reference](inference/docs/api.md) describes the interface between them; other
clients can use it too.

## Repository layout

```text
nWorld/
├── LICENSE
├── CONTRIBUTING.md
├── SECURITY.md
├── inference/            Murea-Fast inference service
│   ├── configs/          service configurations and an example request
│   ├── docs/             deployment, API, performance, provenance
│   ├── scripts/          CLI wrapper, connector merge, acceptance run
│   ├── src/murea_fast/   launcher and client
│   ├── tests/
│   └── vendor/sglang/    pinned SGLang with the model implementation
└── agent/                Murea Agent
    ├── murea_agent/      the agent (Python package)
    ├── app/              FastAPI server
    ├── frontend/         React UI (prebuilt in frontend/dist)
    ├── docs/             co-op mode design notes
    └── tests/
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the checks each part needs before a pull request.

## Security

Neither the inference service nor the agent is meant to be exposed to the internet as is.
Read [SECURITY.md](SECURITY.md) before deploying them on a shared network.

## License

The code is released under the [Apache License 2.0](LICENSE). The vendored SGLang in
`inference/vendor/` keeps its own copyright and license notices, including the files it
carries under MIT and BSD-3-Clause. The Murea-Fast model weights are a derivative of LTX-2.5
and are governed by the
[LTX-2.x Community License Agreement](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x);
see [Provenance](inference/docs/provenance.md) for details. The agent calls third-party
language and image model APIs, which have their own terms.
