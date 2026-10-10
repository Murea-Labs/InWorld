# Contributing

**English** | [中文](CONTRIBUTING_zh.md)

nWorld has two parts, each with its own checks. Run the checks of the part you changed, from
its directory, before sending a pull request. CI runs the same checks when that directory
changes.

## Murea Agent (`agent/`)

See the [agent contributing guide](agent/CONTRIBUTING.md): ruff, the offline tests, the
front-end type check, lint and tests, and the committed front-end build.

## Murea-Fast (`inference/`)

From `inference/`, with the standard library only:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 -m compileall -q src scripts tests
bash -n scripts/murea-fast
```

A change to the vendored engine (`inference/vendor/sglang/`) also needs:

- the change notice in the first line of every file you modify
  (`# Modified by the nWorld authors from SGLang; see inference/docs/provenance.md.`), and a
  row for it in [Provenance](inference/docs/provenance.md) and its Chinese version, with the
  file counts there kept up to date;
- the vendored tests of the code you changed, such as
  `test/unit/test_ltx2_ordered_multiref.py` under `vendor/sglang/python/sglang/multimodal_gen/`,
  run with `pytest` in the full environment ([Deployment](inference/docs/deployment.md#environment)),
  and `tests/smoke_runtime.py` on a GPU host. CI does not run these.

A change that can affect the generated video, such as sampling, attention, the VAE or the
continuation path, also needs a run on a GPU. Say in the pull request which configuration
you used (for example the [acceptance run](inference/docs/deployment.md#acceptance-run)) and
what changed in the output and the timings.

## Documentation

Every document has an English and a Chinese version (`X.md` and `X_zh.md`). Change both in
the same pull request.

## Security

Report vulnerabilities privately, as described in [SECURITY.md](SECURITY.md), not in a public
issue.

## License

Contributions are released under the [Apache License 2.0](LICENSE), like the rest of the code.
