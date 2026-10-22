# FLASH NCCL probe shared-memory ABI

The probe maps `/dev/shm/flash_${FLASH_RUN_ID}_${RANK}`. `FLASH_RUN_ID` must contain only ASCII letters, digits, `_`, `-`, or `.`. Rank comes from `RANK`, then `OMPI_COMM_WORLD_RANK`, `SLURM_PROCID`, `PMI_RANK`, or `LOCAL_RANK`; it defaults to `0`. Set a unique run ID for each training run. The file stays on disk after the process exits so the controller can collect it.

ABI version 1 uses little-endian 64-bit Linux layout. The mapping contains a 64-byte header followed by 8192 records of 128 bytes each, for a total size of 1,048,640 bytes. The C++ definitions use explicit widths and static offset checks. Read the header's `version`, `header_size`, `record_size`, and `capacity` before parsing records.

## Header (64 bytes)

| Offset | Type | Field |
| ---: | --- | --- |
| 0 | `char[8]` | Magic `FLASHNCL` |
| 8 | `u32` | ABI version, currently `1` |
| 12 | `u32` | Header size, `64` |
| 16 | `u32` | Record size, `128` |
| 20 | `u32` | Ring capacity, `8192` |
| 24 | `u32` | Control mode: `0` metadata only; `1` CUDA event timing |
| 28 | `i32` | Global rank from the environment |
| 32 | `u64` | `write_seq`, highest committed record sequence |
| 40 | `u64` | `dropped`, total records overwritten in the ring |
| 48 | `u64` | Heartbeat in nanoseconds from `CLOCK_MONOTONIC` |
| 56 | `u32` | Process ID |
| 60 | `u32` | Header flags; bit 0 means the ring has overflowed |

The probe creates the file on its first intercepted NCCL call. A new header reads its initial mode from `FLASH_PROBE_MODE` (`0` by default); a valid existing header retains its mode. The controller switches modes by writing a 32-bit value at byte offset 24. The probe reads this word atomically at API entry. Mode 0 still records call metadata and never creates CUDA events or synchronizes streams.

## Record (128 bytes)

| Offset | Type | Field |
| ---: | --- | --- |
| 0 | `u64` | Sequence/commit marker, starting at 1 |
| 8 | `u64` | Host `CLOCK_MONOTONIC` start timestamp, ns |
| 16 | `u64` | Host `CLOCK_MONOTONIC` end timestamp, ns |
| 24 | `u64` | CUDA event elapsed duration, ns; 0 when unavailable |
| 32 | `u64` | FNV-1a 64-bit hash of `ncclUniqueId` |
| 40 | `u64` | NCCL count argument |
| 48 | `u64` | Group ID; 0 for an ungrouped call |
| 56 | `i32` | Raw `ncclDataType_t` value |
| 60 | `i32` | Raw `ncclRedOp_t` value; `-1` when the API has no redop |
| 64 | `i32` | Operation: allreduce=1, allgather=2, reducescatter=3, broadcast=4, reduce=5, send=6, recv=7 |
| 68 | `i32` | Peer for send/recv or root for broadcast/reduce; `-1` otherwise |
| 72 | `i32` | Global rank from the environment |
| 76 | `i32` | Communicator size from `ncclCommInitRank`; `-1` if unknown |
| 80 | `i32` | Communicator local rank from `ncclCommInitRank`; `-1` if unknown |
| 84 | `i32` | Operation wrapper result (`ncclResult_t`) |
| 88 | `u32` | Zero-based operation index within a group; 0 for standalone calls |
| 92 | `u32` | Operation count in the group; 1 for standalone calls |
| 96 | `u32` | Zero-based stream index within a group; 0 for standalone calls |
| 100 | `u32` | Distinct stream count in the group; 1 for standalone calls |
| 104 | `u32` | Record flags |
| 108 | `i32` | Actual outer `ncclGroupEnd` result; 0 for standalone calls |
| 112 | `u8[16]` | Reserved, zero-filled |

Record flag bits are: bit 0 grouped call, bit 1 CUDA timing requested, bit 2 CUDA timing valid, bit 3 CUDA event or synchronization error. Group records use one event pair per distinct CUDA stream. The probe records each stream's stop event only after the real outer `ncclGroupEnd` returns, then synchronizes the stop events. `cuda_duration_ns` is the maximum elapsed duration across the group's stream pairs and is copied to every operation record in that group. `start_ns` and `end_ns` are shared host monotonic boundaries from the first operation to the end of timing synchronization. Mode 0 group timestamps cover host submission from the first group operation through the real `GroupEnd` return. For standalone calls, mode 1 places an event pair around the call, waits for its stop event, and stores that event duration; mode 0 stores metadata and host call boundaries only.

`count` is the API's literal count argument: AllGather's per-rank input count and ReduceScatter's per-rank output count. Multiple operation records in one communicator remain available for audit; controller profile statistics can aggregate by `comm_hash`.

## Ring reading

`write_seq` is a cumulative committed-record count and readers must never reset it. Sequence `s` is stored at byte offset `64 + ((s - 1) % capacity) * 128`. After reading a slot, confirm its `seq` equals the requested sequence; a zero or changed value means the slot is being written or was overwritten. The oldest possibly retained sequence is `max(1, write_seq - capacity + 1)`. `dropped` is the cumulative overwrite count. A reader should snapshot `write_seq`, read the corresponding sequence range, then snapshot again if it needs to detect concurrent writes.

For Python, the record layout can be unpacked with `struct.unpack_from("<7Q8i5Ii16s", mmap_obj, offset)`. The header uses `struct.unpack_from("<8s5Ii3QII", mmap_obj, 0)`.
