"""agi2 Modal harness.

``core/``       reusable Modal-first infrastructure (images, Volumes, Dicts, secrets,
                model store, lifecycle, monitoring) plus ``core/runtime/``: Modal-free
                pieces any workload uses (SGLang server, telemetry, system stats).
``workloads/``  one folder per goal; ``arc_eval`` reproduces ARC Prize's Qwen3.8-27B
                results. Each workload's ``runtime/`` stays Modal-free so Kaggle can
                import it.

This package must stay importable without ``modal`` installed: nothing at
package level imports it.
"""
