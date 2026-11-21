"""Allowlist versioned table feature adapters; reject unknown or modified semantics."""
import importlib
from orca_table_task import digest


def adapter(manifest):
    name=manifest.get('table_adapter_module','orca_table_policy')
    if name not in {'orca_table_policy','orca_table_world_policy','orca_table_initial_policy'}:raise ValueError('Unknown table representation adapter')
    module=importlib.import_module(name)
    if module.REPRESENTATION!=manifest['table_representation'] or digest(module.__file__)!=manifest['table_adapter_sha256']:raise ValueError('Table representation identity changed')
    return module
