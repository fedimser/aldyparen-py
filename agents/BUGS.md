## Serious findings

4. **High — deep-zoom arithmetic crashes above 128 decimal digits**
   - `hpn.py:61-68` correctly creates `digits` for scalar operands, but then incorrectly accesses `other.digits`.
   - Any scalar operation on an `Hpn` whose precision exceeds the default 16 groups fails.
   - Confirmed:
     - `Hpn` with precision 18 multiplied by `0.5` raises `AttributeError`.
     - Deep-zoom `graphics.py:129-134` consequently crashes.
   - This affects advertised arbitrary-precision panning and `mixing.py:60-61`, including deep-zoom animations.

5. **High — project saves can destroy the previous valid file**
   - `app.py:276-294` truncates the destination before serialization finishes.
   - Disk-full, I/O errors, process termination, or serialization failure can leave the only project file empty or partially written.
   - Saving should use a temporary file, flush/fsync it, and atomically replace the destination.
