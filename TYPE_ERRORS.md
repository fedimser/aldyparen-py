### Remaining 18 errors

| Category | Count | Description |
|---|---:|---|
| Numba typing deficiencies | **6** | Pyright reports `numba.types` as private even though it is used as part of Numba runtime signatures. |
| Renderer state narrowing | **2** | `frame_rendered` is optional because it begins as `None`; Pyright cannot derive the cross-thread state invariants before it is accessed. |
| Painter type narrowing | **8** | `mix_painters` checks runtime classes, but Pyright does not narrow the two `Painter` arguments to matching concrete painter types. |
| Isolated correctness/typing issues | **2** | Raising a string and calling `decode` on the string returned by `tokenize.untokenize()`. |

All PyQt5 typing deficiencies have been resolved through explicit singleton narrowing and typed enum classes.