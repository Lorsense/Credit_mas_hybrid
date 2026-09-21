"""agent_system package.

Marking this directory as a regular package (rather than an implicit namespace
package) ensures that ``agent_system`` and all of its submodules resolve to THIS
checkout only. Without this file, Python merges every ``agent_system`` directory
found on ``sys.path`` (PEP 420 namespace packages), which on this server also
pulls in the stale sibling ``code/drmas/agent_system`` via the editable ``verl``
``.pth`` install. That merge risks Ray workers importing old source for any
module not present in both trees — the exact "driver/worker import different
source" hazard called out in the S0 gate of SEARCH_PARALLEL_EXPERIMENT_GUIDE.md.
"""
