### Remaining 67 errors

| Category | Count | Description |
|---|---:|---|
| Painter architecture | **43** | Concrete painters do not inherit from or structurally satisfy `Painter`. This affects `Frame`, mixing, presets, and many tests. Fixing it requires reorganizing the painter base class or introducing a proper protocol. |
| PyQt5 typing deficiencies | **13** | PyQt stubs treat `QThreadPool.globalInstance()` and `QApplication.style()` as optional and omit enum attributes such as `ShiftModifier`, `SP_DirIcon`, and `SP_FileIcon`. |
| Numba typing deficiencies | **6** | Pyright reports `numba.types` as private even though it is used as part of Numba runtime signatures. |
| Renderer state narrowing | **2** | `frame_rendered` is optional because it begins as `None`; Pyright cannot derive the cross-thread state invariants before it is accessed. |
| Isolated correctness/typing issues | **3** | Raising a string, optional test error text, and the conditional return type of `tokenize.untokenize()`. |

The dominant remaining problem is the painter hierarchy, beginning with `__init__.py` and propagating through `mixing.py`, `graphics.py`, and their tests.