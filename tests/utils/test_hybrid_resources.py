"""CPU-only tests of the actual resource methods with a deferred Ray mock."""
import ast
import copy
import importlib.util
import logging
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("hybrid_resources_test", ROOT / "verl/utils/credit_resources.py")
resources = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resources)


def method(class_name, method_name, namespace):
    path = ROOT / "verl/single_controller/ray/base.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    function.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), "exec"), namespace)
    return namespace[method_name]


def test_actual_actor_layout_and_value_reservation():
    config = {"nnodes": 2, "n_gpus_per_node": 8, "agent_gpus_per_node": [7, 8]}
    assert resources.agent_gpu_layout(config) == [7, 8]
    assert resources.agent_world_size(config) == 15
    nodes = {"value": {"GPU": 8, "CPU": 32}, "actors": {"GPU": 8, "CPU": 32}}
    resources.validate_pool_layout(nodes, {"actors": [7, 8]}, "value", cpus_per_gpu=1, reserved_cpus=2)
    with pytest.raises(ValueError):
        resources.validate_pool_layout(nodes, {"actors": [8, 8]}, "value", cpus_per_gpu=1, reserved_cpus=2)
    with pytest.raises(ValueError):
        resources.agent_gpu_layout({**config, "agent_gpus_per_node": [15]})
    with pytest.raises(ValueError):
        resources.agent_gpu_layout({**config, "agent_gpus_per_node": [7, 9]})


def test_placement_reserves_largest_before_submitting_smaller_and_preserves_order():
    available = [8, 7]  # Adversarial first-fit scheduler: smaller first would block.
    events = []

    def create(bundles, strategy, name, lifetime):
        count = len(bundles)
        events.append(("create", count))
        pg = SimpleNamespace(id=name, bundle_count=count)

        def reserve():
            node = next((i for i, free in enumerate(available) if free >= count), None)
            assert node is not None, "placement would wait forever on this layout"
            available[node] -= count
            events.append(("ready", count))
            return pg

        pg.ready = lambda: reserve
        return pg

    namespace = {"placement_group": create, "ray": SimpleNamespace(get=lambda pending: pending())}
    allocate = method("RayResourcePool", "get_placement_groups", namespace)
    pool = SimpleNamespace(pgs=None, name_prefix="hybrid", _store=[7, 8], max_colocate_count=1,
                           use_gpu=True, accelerator_type=None, detached=False)
    groups = allocate(pool)
    assert events == [("create", 8), ("ready", 8), ("create", 7), ("ready", 7)]
    assert [pg.bundle_count for pg in groups] == [7, 8]
    assert allocate(pool) is groups  # Existing groups are never reserved twice.


@pytest.mark.parametrize("layout,bundles", [([7, 8], [7, 8]), ([7, 8], [8, 8]), ([15], [15])])
def test_worker_rank_assignment_tracks_each_group_after_node_ip_sort(layout, bundles):
    groups = [SimpleNamespace(id=str(i), bundle_count=count) for i, count in enumerate(bundles)]
    created = []

    class WorkerFactory:
        cls = object

        def update_options(self, options):
            self.options = copy.deepcopy(options)

        def __call__(self, **kwargs):
            created.append((kwargs["placement_group"].id, copy.deepcopy(self.options["runtime_env"]["env_vars"])))
            return object()

    register = SimpleNamespace(get_rank_zero_info=SimpleNamespace(remote=lambda: {"MASTER_ADDR": "localhost", "MASTER_PORT": "1234"}))
    namespace = {
        "sort_placement_group_by_node_ip": lambda pgs: list(reversed(pgs)),
        "ray": SimpleNamespace(get=lambda result: result, get_actor=lambda name: register),
        "list_named_actors": lambda **kwargs: ["hybrid_register_center"],
        "time": time, "logging": logging,
    }
    initialize = method("RayWorkerGroup", "_init_with_resource_pool", namespace)
    pool = SimpleNamespace(use_gpu=True, world_size=sum(layout), max_colocate_count=1, store=layout,
                           get_placement_groups=lambda **kwargs: groups)
    worker = SimpleNamespace(device_name="cuda", name_prefix="hybrid", _workers=[], _worker_names=[],
                             _ray_wait_register_center_timeout=1)
    initialize(worker, pool, WorkerFactory(), True, False)
    assert len(created) == sum(layout)
    assert [int(env["RANK"]) for _, env in created] == list(range(sum(layout)))
    assert all(int(env["WORLD_SIZE"]) == sum(layout) for _, env in created)
    for group, expected_count in enumerate(layout):
        local = [env for pg_id, env in created if pg_id == str(group)]
        assert len(local) == expected_count
        assert [int(env["RAY_LOCAL_RANK"]) for env in local] == list(range(expected_count))
        assert all(int(env["RAY_LOCAL_WORLD_SIZE"]) == expected_count for env in local)
