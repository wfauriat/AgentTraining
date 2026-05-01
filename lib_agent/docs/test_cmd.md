# Manual REPL test prompts — bash tools

Run from `lib_agent/`:

```bash
.venv/bin/python chat.py
```

Type each prompt at `You: `. For destructive tools you'll see `approve? [y/N]` — answer `y` to allow, `n` to deny. Type `quit` when done.

---

## Read-only (no HITL gate)

**1. List the workspace root**
```
List the contents of the workspace root.
```

**2. Find markdown files**
```
Find all *.md files anywhere in the workspace.
```

**3. List a subdirectory**
```
List the contents of subdir/ in the workspace.
```

**4. Find with depth**
```
Find every file in the workspace whose name contains 'cycle'.
```

---

## Mutations (mkdir + delete) — `delete_file` triggers HITL

**5. Make a directory**
```
Make a directory called notes in the workspace, then list the workspace to confirm.
```

**6. Write, list, then delete (two HITL gates: write + delete)**
```
Write 'gone soon' into 'tmp.txt', list the workspace, then delete tmp.txt and list again.
```
Approve both gates with `y`.

**7. Try the deny path**
```
Delete the file 'hello.txt' from the workspace.
```
Answer `n` at the gate. The file should NOT be removed — verify with the next prompt.

```
List the workspace root.
```

---

## Sandbox defenses (negative tests)

**8. ls outside the workspace — should be rejected**
```
Call list_directory with path='../../../etc' as a sandbox test. Report the exact error.
```

**9. delete outside the workspace — gate fires (auto-approve), tool rejects**
```
Call delete_file with path='../../../tmp/anything' as a sandbox test. Report the error message.
```
Approve with `y` — the sandbox itself rejects after the gate.

**10. mkdir outside — should be rejected**
```
Try to make a directory at '../../../tmp/should_fail' as a sandbox test. Report whether it worked.
```

---

## Multi-agent (try `--multi` flag)

```bash
.venv/bin/python chat.py --multi
```

**11. Single-worker code path**
```
Find all the .md files in the workspace, then count how many there are.
```

**12. Cross-worker — research then file write (2 HITL gates)**
```
Look up what photosynthesis is in the local corpus, then save a one-sentence summary to 'photo_note.txt' in the workspace.
```

**13. Multi-step file ops**
```
Make a directory called 'reports', write 'first line' into reports/log.md, then list the contents of reports/.
```

---

## What to watch in Phoenix

Open http://localhost:6006, project `lib_agent`. After each turn:
- Tool spans appear with their args + result
- HITL approvals show as paused/resumed nodes
- Multi-agent runs show a hierarchy: supervisor → research_agent / code_agent → tool calls inside
