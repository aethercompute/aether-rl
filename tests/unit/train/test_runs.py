from types import SimpleNamespace

from aether_rl.trainer.runs import MultiRunManager


def test_discover_runs_preserves_dots_in_directory_name(tmp_path):
    run_id = "run_deepscaler-r1-qwen-1.5b-stage1"
    (tmp_path / run_id).mkdir()
    discovered = []

    manager = object.__new__(MultiRunManager)
    manager.world = SimpleNamespace(is_master=True)
    manager.output_dir = tmp_path
    manager.id_2_idx = {}
    manager.unused_idxs = {0}
    manager._forgotten_hooks = []
    manager.get_orchestrator_config = lambda candidate: discovered.append(candidate)

    manager.discover_runs()

    assert discovered == [run_id]
