"""scripts/run_signnet.sh ("Train the cutter") with a stand-in GPU environment: the epochs reach signnet.log as they
come (PYTHONUNBUFFERED), and a NAME tried again keeps its old log apart (signnet.log.1), so epochs never mix."""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# The stand-in nnslr: says whether Python would buffer, and prints an epoch line when it trains.
FAKE_NNSLR = '''#!/bin/sh
echo "nnslr $1 PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-unset}"
case " $* " in
  *" --dry-run "*) ;;
  *" train-signnet "*|"train-signnet "*) echo "signnet epoch 1/40 train_loss 4.0000 val_recall 0.100 val_false_per_frame 4.90 80s" ;;
esac
exit 0
'''


def run_script(tmp_path: Path, name: str = 'n1') -> subprocess.CompletedProcess:
    tools, venv = tmp_path / 'bin', tmp_path / 'venv' / 'bin'
    for folder in (tools, venv):
        folder.mkdir(parents=True, exist_ok=True)
    for tool, text in ((tools / 'nvidia-smi', '#!/bin/sh\nexit 0\n'), (venv / 'python', '#!/bin/sh\nexit 0\n'),
                       (venv / 'nnslr', FAKE_NNSLR)):
        tool.write_text(text)
        tool.chmod(0o755)
    return subprocess.run(['bash', str(REPO / 'scripts' / 'run_signnet.sh'), name], capture_output=True, check=False,
                          text=True, timeout=60, env={'PATH': f'{tools}:/usr/bin:/bin',
                                                      'NNSLR_DATA_ROOT': str(tmp_path / 'data'),
                                                      'NNSLR_GPU_VENV': str(venv.parent)})


def test_the_training_is_unbuffered_and_logged(tmp_path):
    process = run_script(tmp_path)
    assert process.returncode == 0, process.stderr + process.stdout
    log = (tmp_path / 'data' / 'runs' / 'n1' / 'signnet.log').read_text()
    assert 'nnslr train-signnet PYTHONUNBUFFERED=1' in log and 'signnet epoch 1/40 ' in log
    assert not (tmp_path / 'data' / 'runs' / 'n1' / 'signnet.log.1').exists()


def test_a_name_tried_again_keeps_its_old_log_apart(tmp_path):
    run = tmp_path / 'data' / 'runs' / 'n1'
    run.mkdir(parents=True)
    (run / 'signnet-best.pt').write_bytes(b'weights of a training stopped half way')
    (run / 'signnet.log').write_text('signnet epoch 30/40 train_loss 1.0 val_recall 0.5 val_false_per_frame 4.0 80s\n')
    (run / 'signnet.log.1').write_text('an older try\n')
    process = run_script(tmp_path)
    assert process.returncode == 0, process.stderr + process.stdout
    log = (run / 'signnet.log').read_text()
    assert 'epoch 30/40' not in log and 'signnet epoch 1/40 ' in log
    assert 'restart from scratch' in log and f'the log of the earlier try: {run}/signnet.log.1' in log
    assert (run / 'signnet.log.1').read_text().startswith('signnet epoch 30/40 ')  # the older one replaced
