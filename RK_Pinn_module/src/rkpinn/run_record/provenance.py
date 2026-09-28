"""Where a thing in the store came from: the commit, the package version and
the machine.

Every number must be traceable (WORKFLOW.md, section 3). Writing to the store
from a working tree with uncommitted changes is refused, unless the caller
asks for it, and then the record says that it cannot be traced.
"""
from __future__ import annotations

import os
import platform
import subprocess
import time

import rkpinn


class UncommittedChanges(RuntimeError):
    """The package has changes that are not committed."""


def folder_of_the_module() -> str:
    """The folder `RK_Pinn_module`, found from where the package is."""
    package = os.path.dirname(os.path.abspath(rkpinn.__file__))      # .../src/rkpinn
    return os.path.dirname(os.path.dirname(package))


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", folder_of_the_module()) + arguments,
        check=True, capture_output=True, text=True).stdout.strip()


def commit_of_the_package() -> str:
    """The commit the package is at."""
    return _git("rev-parse", "HEAD")


def uncommitted_changes() -> list[str]:
    """The files of the module that are changed or new and not committed."""
    listing = _git("status", "--porcelain", "--", folder_of_the_module())
    return [line for line in listing.splitlines() if line.strip()]


def processor_of_this_machine() -> str:
    try:
        with open("/proc/cpuinfo") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def provenance(allow_uncommitted_changes: bool = False) -> dict:
    """The record written beside everything the package puts in the store."""
    changes = uncommitted_changes()
    if changes and not allow_uncommitted_changes:
        raise UncommittedChanges(
            "the package has uncommitted changes, so what it writes could not be "
            "traced to a commit. Commit them, or allow it and have the record say "
            "so. Changed: %s" % "; ".join(changes))
    return {
        "package_version": rkpinn.__version__,
        "commit": commit_of_the_package(),
        "traceable_to_the_commit": not changes,
        "uncommitted_changes": changes,
        "machine": platform.node(),
        "processor": processor_of_this_machine(),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
