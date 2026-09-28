# The experiments table on the Notion project page

**Last updated:** 2026-09-28 · **State:** defined, not created

An inline database on the project page. One row per experiment. It answers two questions at a glance: which experiments exist, and where the write-up of each is.

It covers experiments started on or after 28 September 2026. Earlier work is not added.

## Columns

| Column | Type | Content |
|---|---|---|
| Experiment | title | its plain English name, the same as its folder |
| Question | text | one sentence |
| State | select | Planned, Running, Analysed, Written up, Stopped |
| Network | select | the name in the configuration |
| Loss | multi-select | the names in the configurations |
| Target | select | the name in the configuration |
| Steps and stages | text | the cells that were run |
| Run keys | text | the keys in the store |
| Folder | text | the experiment's folder in the repository |
| Commit | text | the commit the runs were trained at |
| Write-up | relation to the write-ups database | the write-up of record |
| To-do | relation to the to-do database | the to-do that holds the worklog |
| Started, finished | dates | |

## States

| State | Means | Set when |
|---|---|---|
| Planned | the question and design are agreed | the folder is made |
| Running | jobs are on the farm | the first job is submitted |
| Analysed | the standard report exists for every run | the last report is produced |
| Written up | a write-up exists and is linked | the write-up is published |
| Stopped | ended without a result | George rules so; the reason goes in Question |

## Rules

1. A row is added when the experiment's folder is made, not when it finishes.
2. The Write-up relation is empty until a write-up exists. A row never points to a draft.
3. A write-up is Provisional until George verifies it. The table does not show trust; the write-up does.
4. A stopped experiment keeps its row.

## Creating it

The table is a new database. It needs one `ADD COLUMN … RELATION` to the write-ups data source `3265d544-b9d9-8000-8b4a-000b13a4b7c6` and one to the to-do data source `e535955f-0756-4222-bed8-47f25e2b020f`. The workspace rules name four canonical databases, so this fifth needs a line in `CLAUDE.md` once it exists.
