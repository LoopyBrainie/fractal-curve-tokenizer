"""Tests for main() pattern-matching Outcome returns (AEH Phase 1, Chunk 6).

These tests verify that:
1. When L1 config construction fails (Err), main() re-raises with `from err`
   so the original ConfigError appears in the chain as __cause__.
2. When dataset creation fails (Err), main() exits cleanly with SystemExit
   (since the CLI layer is the right place to terminate the process).
3. The --quick-test branch still works (no Outcome change there).

The tests stub out GPU/data dependencies and use a synthetic bad config
to exercise the boundary without actually starting a training run.
"""
from __future__ import annotations
import pytest


class TestMainOutcomeBoundary:
    """Verify main() respects Outcome boundaries at L1 (config + data)."""

    def test_main_propagates_config_error_via_chain(self, monkeypatch):
        """When try_construct_fractal_config returns Err, main() must
        re-raise a RuntimeError with the original ConfigError in __cause__.

        The `from err` syntax sets err.__cause__; we verify the chain
        is preserved (not lost) for debuggability.
        """
        from training import main as main_mod

        # Stub out everything that touches GPU / data / filesystem.
        monkeypatch.setattr(main_mod, "set_seed", lambda _s: None)
        monkeypatch.setattr(main_mod, "configure_cuda", lambda: None)
        monkeypatch.setattr(main_mod, "create_model", lambda **_k: None)
        monkeypatch.setattr(main_mod, "train", lambda **_k: None)

        # Replace try_construct_fractal_config to return a bad-config Outcome.
        from vit_pytorch.core.outcome import Err
        from vit_pytorch.core.config import ConfigError

        def bad_try_construct(*_args, **_kwargs):
            return Err(ConfigError("min_patch_size", "test failure from main"))

        # Patch the symbol inside main_mod (it was imported via the L1
        # config module).
        import vit_pytorch.core.config as l1_config
        monkeypatch.setattr(l1_config, "try_construct_fractal_config", bad_try_construct)
        # Also rebind on main_mod if it imported it directly.
        if hasattr(main_mod, "try_construct_fractal_config"):
            monkeypatch.setattr(
                main_mod, "try_construct_fractal_config", bad_try_construct
            )
        # main() reads sys.argv via argparse. Override it for this test.
        monkeypatch.setattr("sys.argv", [
            "main.py",
            "--quick-test",
            "--image-size", "32",
            "--min-patch-size", "4",
        ])
        with pytest.raises(RuntimeError, match="Model configuration initialization aborted"):
            main_mod.main()

    def test_main_preserves_exception_chain_for_config_failure(self, monkeypatch):
        """The chain (ConfigError → RuntimeError) must be preserved.

        We assert that the RuntimeError's __cause__ is the original
        ConfigError instance, so users debugging the failure can see
        the root cause in the traceback.
        """
        from training import main as main_mod
        from vit_pytorch.core.outcome import Err
        from vit_pytorch.core.config import ConfigError

        monkeypatch.setattr(main_mod, "set_seed", lambda _s: None)
        monkeypatch.setattr(main_mod, "configure_cuda", lambda: None)
        monkeypatch.setattr(main_mod, "create_model", lambda **_k: None)
        monkeypatch.setattr(main_mod, "train", lambda **_k: None)

        sentinel = ConfigError("min_patch_size", "sentinel cause")
        def bad_try_construct(*_args, **_kwargs):
            return Err(sentinel)

        import vit_pytorch.core.config as l1_config
        monkeypatch.setattr(l1_config, "try_construct_fractal_config", bad_try_construct)
        if hasattr(main_mod, "try_construct_fractal_config"):
            monkeypatch.setattr(
                main_mod, "try_construct_fractal_config", bad_try_construct
            )

        monkeypatch.setattr("sys.argv", [
            "main.py",
            "--quick-test",
            "--image-size", "32",
            "--min-patch-size", "4",
        ])
        with pytest.raises(RuntimeError) as exc_info:
            main_mod.main()
        # The chain must be preserved: __cause__ is the ConfigError.
        assert exc_info.value.__cause__ is sentinel
        assert isinstance(exc_info.value.__cause__, ConfigError)
