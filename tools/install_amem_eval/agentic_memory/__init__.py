"""SDK-compatible shim around the WujiangXu/AgenticMemory eval repo.

The original SDK at agiresearch/A-mem exposes ``AgenticMemorySystem``
via ``agentic_memory.memory_system``. The eval repo (this distribution)
keeps the same class but adds methods used by the A-mem LoCoMo
evaluation (``find_related_memories_raw``, etc.). The shim re-exports
under the SDK path so importers don't have to change.
"""
