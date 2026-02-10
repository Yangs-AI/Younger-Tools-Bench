# Younger-Tools-Bench

Younger-Tools-Bench provides benchmarking utilities for Younger datasets and related workflows.

## Optional Dependencies (Extras)
Install an extra with:

```
pip install "younger-tools-bench[EXTRA_NAME]"
```

Available extras:

- `construct`: Graph construction utilities based on torch-geometric.
- `develop`: Developer tools (docs, pytest, release tooling).

## Contributing
This repository is a submodule of the parent project. For the canonical workflow and submodule pointer rules, see [CONTRIBUTING.md](https://github.com/Yangs-AI/Younger/CONTRIBUTING.md).

If you are working inside a parent clone:
- Commit changes in this submodule repo and open a PR to the submodule upstream.
- Do not update the parent repo submodule pointer unless a maintainer asks; mention any required pointer bump in your PR description.

If you are working standalone:
- Fork this repo, add `upstream`, create a feature branch, then open a PR back to upstream.

### Git workflow (sync with upstream)
```bash
# 1) Sync with upstream
git remote -v               # confirm origin/upstream are set correctly
git fetch --prune upstream   # sync upstream refs and drop deleted branches
git checkout dev             # ensure you are on the main dev branch
git pull --rebase upstream dev  # replay local dev on top of upstream/dev

# If you keep local commits on dev, rebase explicitly:
git rebase upstream/dev      # same as above, but explicit and easier to debug
# If conflicts happen:
git status                   # see which files are conflicted
# resolve files, then
git add <conflicted-file>    # mark conflicts as resolved
git rebase --continue        # continue replaying commits
# to abort:
git rebase --abort           # roll back to pre-rebase state

# 2) Create a feature branch
git checkout -b feat/my-change

# 3) Make changes and commit
git add .
git commit -m "feat: my change"

# 4) Push to your fork
git push origin feat/my-change
```

### Fixing edits made on a detached HEAD
If you forgot to switch branches and edited files on a `Previous HEAD position`, stash the changes, switch to the target branch, then restore them:

```bash
# 1) Stash the current changes
git stash push -m "temp: bench local edits"

# 2) Switch to dev
git checkout dev

# 3) Restore the changes
git stash pop
```

### Local development
From this repo root:

```bash
# 1) Install with dev tools
pip install -e .[develop]

# 2) Run tests (if present)
pytest tests
```

### Common commands
```bash
younger-tools-bench --help
```

### Developer extras
This module uses the `develop` extra for docs tooling:

```bash
pip install -e .[develop]
```

### More docs
- [Homepage](https://younger.yangs.ai/tools/bench)
