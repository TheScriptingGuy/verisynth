# Releasing to PyPI

Two distributions are published from this repo:

| Distribution        | Source                        | Built by                          |
| ------------------- | ----------------------------- | --------------------------------- |
| `verisynth`         | pure-Python (`verisynth/`)    | `python -m build` (sdist + wheel) |
| `verisynth_kernels` | PyO3/maturin (`rust/`)        | `maturin` (per-platform wheels)   |

Both are published to **PyPI** on every published GitHub Release, using
**Trusted Publishing (OIDC)** — no API tokens or passwords are stored in the
repository or in GitHub secrets.

## One-time setup on PyPI (per package)

Trusted Publishing must be configured once for each package on PyPI. Do this
while signed in to your PyPI account (this is the only step that needs your
PyPI login):

1. Go to <https://pypi.org/manage/account/publishing/> (the "Publishing" page).
2. Under **Add a new pending publisher**, add one entry for **each** package:

   **`verisynth`**
   - PyPI Project Name: `verisynth`
   - Owner: `TheScriptingGuy`
   - Repository name: `verisynth`
   - Workflow name: `publish-python.yml`
   - Environment name: `pypi`

   **`verisynth_kernels`**
   - PyPI Project Name: `verisynth_kernels`
   - Owner: `TheScriptingGuy`
   - Repository name: `verisynth`
   - Workflow name: `publish-kernels.yml`
   - Environment name: `pypi`

   (Use a *pending* publisher if the project doesn't exist on PyPI yet — the
   first successful upload creates it.)

## One-time setup on GitHub

Create an environment named **`pypi`** in the repo
(**Settings → Environments → New environment**). The publish jobs run in this
environment; you may optionally add required reviewers there to gate releases.

## Cutting a release

1. Bump the version in **both** `pyproject.toml` and `rust/Cargo.toml`
   (and `rust/pyproject.toml`) so the two distributions share a version.
2. Commit, tag, and push:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```
3. Publish a **GitHub Release** for that tag (Releases → Draft a new release).
   Publishing the release triggers both `publish-python.yml` and
   `publish-kernels.yml`, which build and upload both distributions to PyPI.

## CI

`ci.yml` runs on every push and pull request: it builds the Rust kernels with
`maturin develop` and runs the full test suite against both the Rust backend
and the pure-numpy reference backend across Python 3.10–3.12.
