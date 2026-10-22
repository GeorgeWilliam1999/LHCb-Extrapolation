#!/bin/bash
# Check the project out at one commit, in a folder of its own, for the jobs of
# the farm to run from. Work on the package then goes on in the repository
# without changing what a job runs, and a job that is started again after its
# time limit runs the code it started with.
#
#   make_code.sh <commit> <folder of the copies>
#
# Prints the folder of the copy. It is a git worktree, so the package finds its
# commit there and finds no uncommitted change.
set -eu
COMMIT=$(git -C "$(dirname "$0")" rev-parse "$1")
COPIES="$2"
PROJECT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
CODE="$COPIES/${COMMIT:0:8}"
if [ ! -d "$CODE" ]; then
    mkdir -p "$COPIES"
    git -C "$PROJECT" worktree add --detach "$CODE" "$COMMIT" >&2
fi
test "$(git -C "$CODE" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$CODE" status --porcelain)"
echo "$CODE"
