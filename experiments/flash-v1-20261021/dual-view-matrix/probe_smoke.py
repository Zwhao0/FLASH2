"""Use the already planned wall smoke with the real measurement probe preload."""
import hashlib
import json
from pathlib import Path
import execute as ex

HERE=Path(__file__).resolve().parent
original_opts=ex.opts
def probe_opts(run,probe=False):
    value=original_opts(run,probe=True)
    assert 'LD_PRELOAD=/workspace/flash2/probe/libflash_nccl_probe.so' in value
    ex.save('probe-smoke-launch-options.json',dict(run=run,environment_options=value,harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    return value
ex.opts=probe_opts
if __name__=='__main__':
    ex.smoke()
