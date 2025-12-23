"""
Shared pytest fixtures for caliby tests
"""

import pytest
import tempfile
import os
import sys

# Add build directory to path for local testing
build_path = os.path.join(os.path.dirname(__file__), '..', 'build')
if os.path.exists(build_path):
    sys.path.insert(0, build_path)


@pytest.fixture(scope="session")
def temp_dir_session():
    """Create a temporary directory for test indexes (session-scoped)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create heapfile in the temp directory
        heapfile = os.path.join(tmpdir, "heapfile")
        open(heapfile, 'w').close()
        
        # Change to temp directory so heapfile is found
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        
        yield tmpdir
        
        os.chdir(old_cwd)


@pytest.fixture(scope="session")
def caliby_module_session(temp_dir_session):
    """Import caliby module once per session (avoids double registration)."""
    import caliby
    # Configure buffer pool via module-level API
    # Note: VIRTGB is auto-computed per-index, only PHYSGB needs to be set
    caliby.set_buffer_config(size_gb=0.3)
    return caliby


# Module-scoped fixtures that reference session fixtures
@pytest.fixture(scope="module")
def temp_dir(temp_dir_session):
    """Module-scoped alias for temp_dir_session."""
    return temp_dir_session


@pytest.fixture(scope="module") 
def caliby_module(caliby_module_session):
    """Module-scoped alias for caliby_module_session."""
    return caliby_module_session
