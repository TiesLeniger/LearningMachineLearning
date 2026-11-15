# External tools

Third-party executables called as subprocesses. **Not committed** (see `.gitignore`): download them yourself from the official sources below.

| Tool  | Version | Source | Local path | SHA-256 (Windows build) |
|-------|---------|--------|------------|-------------------------|
| XFOIL | 6.99    | https://web.mit.edu/drela/Public/web/xfoil/ | `tools/xfoil/xfoil.exe` | `c17342f84ae260c2b11a74cd0e2fb8189a5f8954c6bb7a8467a0f27055c7faea` |
| AVL   | 3.40    | https://web.mit.edu/drela/Public/web/avl/   | `tools/avl/avl.exe`     | `88a84deca2daeb8ef3a0407520814e941c9bfdd5cdd8a2de450d122a88b4f42d` |

Both are by Mark Drela (MIT) and licensed under the GNU GPL.

## Lookup order
Code resolves each executable as:
1. Environment variable `XFOIL_EXE` / `AVL_EXE` (e.g. on the tower or Linux)
2. `tools/<tool>/<tool>.exe` in this repo
3. `PATH`

Tests that need a tool are skipped when it is not found (e.g. on CI).

## Verify a download
```
certutil -hashfile tools\xfoil\xfoil.exe SHA256
```
Compare with the table. A different hash means a different build: results may differ.
