## Serious findings

1. **Critical — clearing a movie can silently discard project data**
   - `main.py:332-335` empties `frames` directly but does not set `have_unsaved_changes` or reset `selected_frame_idx`.
   - Confirmed runtime state after clearing: `frames=[]`, `selected_frame_idx=0`, `have_unsaved_changes=False`.
   - A user can clear a previously saved movie and exit without an unsaved-changes warning.

3. **High — asynchronous preview races with frame deletion**
   - `async_runners.py:33-36` indexes the current frame after rendering without checking whether the movie was cleared or changed meanwhile.
   - Confirmed with an emptied frame list: `IndexError: list index out of range`.
   - The same access treats index `-1` as the last frame, so invalid selection state can also identify the wrong frame.

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

6. **High — large allowed video resolutions divide by zero**
   - `video.py:30-33` permits `frames_per_part == 0` when one RGB frame exceeds 100 MB, then divides by it.
   - The UI permits dimensions up to `10000 × 10000` in `main.xml:793-827`, well beyond that threshold.
   - Such video renders fail immediately rather than splitting or rejecting the resolution.

7. **High — video resources and temporary files are not safely cleaned up**
   - `video.py:62-69` never closes `VideoFileClip` instances or the concatenated clip.
   - Exceptions or cancellation also bypass temporary-part cleanup.
   - This can exhaust file descriptors and prevents temporary-file deletion on platforms that lock open files.

