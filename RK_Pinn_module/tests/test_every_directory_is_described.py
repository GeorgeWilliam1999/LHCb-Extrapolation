"""Gate: every directory is described, and every description is true.

The rule (WORKFLOW.md, section 5): every directory of the module holds a
README.md whose Contents table names every file and subdirectory in it, and
says whether each is built or planned.

This test fails when
  * a directory has no README.md;
  * a file or subdirectory exists that its README.md does not name;
  * something marked "built" does not exist;
  * something marked "planned" exists (its state was not updated).
"""
from __future__ import annotations

import os
import re

import pytest

MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IGNORED_NAMES = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".git", ".gitignore"}
IGNORED_ENDINGS = (".pyc", ".egg-info")

ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|.*\|\s*(built|planned)\s*\|\s*$")


def is_ignored(name: str) -> bool:
    return name in IGNORED_NAMES or name.endswith(IGNORED_ENDINGS)


def directories():
    found = []
    for here, subdirectories, _ in os.walk(MODULE_ROOT):
        subdirectories[:] = sorted(d for d in subdirectories if not is_ignored(d))
        found.append(here)
    return found


def described(readme_path: str) -> dict[str, str]:
    """Name -> state, read from the Contents table of one README."""
    rows = {}
    with open(readme_path) as handle:
        for line in handle:
            match = ROW.match(line.rstrip("\n"))
            if match:
                rows[match.group(1).rstrip("/")] = match.group(2)
    return rows


def relative(path: str) -> str:
    return os.path.relpath(path, MODULE_ROOT)


@pytest.mark.parametrize("directory", directories(), ids=relative)
def test_directory_has_a_description(directory):
    assert os.path.isfile(os.path.join(directory, "README.md")), (
        f"{relative(directory)} has no README.md")


@pytest.mark.parametrize("directory", directories(), ids=relative)
def test_description_names_everything_present(directory):
    readme = os.path.join(directory, "README.md")
    if not os.path.isfile(readme):
        pytest.skip("no README.md; reported by the test above")
    rows = described(readme)
    present = sorted(n for n in os.listdir(directory) if not is_ignored(n))
    missing = [n for n in present if n not in rows]
    assert not missing, (
        f"{relative(readme)} does not describe: {', '.join(missing)}")


@pytest.mark.parametrize("directory", directories(), ids=relative)
def test_stated_state_is_true(directory):
    readme = os.path.join(directory, "README.md")
    if not os.path.isfile(readme):
        pytest.skip("no README.md; reported by the test above")
    wrong = []
    for name, state in described(readme).items():
        exists = os.path.exists(os.path.join(directory, name))
        if state == "built" and not exists:
            wrong.append(f"{name} is marked built but does not exist")
        if state == "planned" and exists:
            wrong.append(f"{name} is marked planned but exists")
    assert not wrong, f"{relative(readme)}: " + "; ".join(wrong)
