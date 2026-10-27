"""Load stdlib before the home-I/O guard inspects a PM Python symlink sibling.

Verification-only pytest plugin: no guard is disabled and no state is accessed.
Use -p tests.hermes_cli.scope_preload with the canonical test runner.
"""
import sysconfig  # noqa: F401
import zoneinfo  # noqa: F401
