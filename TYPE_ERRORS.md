### Remaining 12 errors

| Category | Count | Description |
|---|---:|---|
| Numba typing deficiencies | **6** | Pyright reports `numba.types` as private even though it is used as part of Numba runtime signatures. |
| Renderer state narrowing | **2** | `frame_rendered` is optional because it begins as `None`; Pyright cannot derive the cross-thread state invariants before it is accessed. |
| Isolated correctness/typing issues | **2** | Raising a string and calling `decode` on the string returned by `tokenize.untokenize()`. |

All PyQt5 typing deficiencies and painter type-narrowing errors have been resolved.