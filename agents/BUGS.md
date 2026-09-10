## Serious findings


5. **High — project saves can destroy the previous valid file**
   - `app.py:276-294` truncates the destination before serialization finishes.
   - Disk-full, I/O errors, process termination, or serialization failure can leave the only project file empty or partially written.
   - Saving should use a temporary file, flush/fsync it, and atomically replace the destination.
