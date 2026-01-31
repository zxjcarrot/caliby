"""
Shared pytest fixtures for caliby tests
"""

import pytest
import tempfile
import os
import sys
import shutil
import glob

# Add build directory to path for local testing
build_path = os.path.join(os.path.dirname(__file__), '..', 'build')
if os.path.exists(build_path):
    # Remove from sys.path if already there to force fresh import
    if build_path in sys.path:
        sys.path.remove(build_path)
    # Insert at the beginning to prioritize local build
    sys.path.insert(0, build_path)


def cleanup_temp_files():
    """Clean up temporary test files and directories."""
    # Save current directory
    try:
        original_cwd = os.getcwd()
    except FileNotFoundError:
        original_cwd = None
    
    # Change to a safe directory before cleanup
    safe_dir = os.path.expanduser('~')
    os.chdir(safe_dir)
    
    # Clean up /tmp/test_* directories
    for pattern in ['/tmp/test_*', '/tmp/caliby_*', '/tmp/pytest-of-*']:
        for path in glob.glob(pattern):
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.remove(path)
            except Exception:
                pass  # Ignore cleanup errors
    
    # Clean up catalog.dat in common locations
    for path in ['/tmp/catalog.dat']:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
    
    # Restore original directory if it still exists
    if original_cwd and os.path.exists(original_cwd):
        os.chdir(original_cwd)


@pytest.fixture(scope="session", autouse=True)
def cleanup_before_session():
    """Clean up before test session starts."""
    cleanup_temp_files()
    yield
    # Clean up after session ends
    cleanup_temp_files()


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
        
        # Clean up catalog.dat in temp directory
        catalog_path = os.path.join(tmpdir, "catalog.dat")
        if os.path.exists(catalog_path):
            try:
                os.remove(catalog_path)
            except Exception:
                pass


@pytest.fixture(scope="session")
def caliby_module_session(temp_dir_session):
    """Import caliby module once per session."""
    # Force fresh import by removing from sys.modules if already loaded
    if 'caliby' in sys.modules:
        del sys.modules['caliby']
    
    import caliby
    # Configure buffer pool via module-level API
    caliby.set_buffer_config(size_gb=1)
    yield caliby
    # Close caliby to ensure proper cleanup
    try:
        caliby.close()
    except Exception:
        pass


# Module-scoped fixtures that reference session fixtures
@pytest.fixture(scope="module")
def temp_dir(temp_dir_session):
    """Module-scoped alias for temp_dir_session."""
    return temp_dir_session


@pytest.fixture(scope="module") 
def caliby_module(caliby_module_session):

    """Module-scoped alias for caliby_module_session."""
    return caliby_module_session


@pytest.fixture(scope="function")
def initialized_db(caliby_module, temp_dir):
    """Function-scoped fixture that provides a fresh caliby instance for each test."""
    # Ensure caliby is opened in the temp directory
    caliby_module.open(temp_dir)
    yield caliby_module
    # Close after each test to ensure cleanup
    try:
        caliby_module.close()
    except Exception:
        pass
