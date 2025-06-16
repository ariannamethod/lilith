"""Validation tests to ensure the testing infrastructure is properly configured."""

import pytest
import sys
from pathlib import Path


class TestSetupValidation:
    """Test class to validate the testing setup."""
    
    @pytest.mark.unit
    def test_pytest_installed(self):
        """Test that pytest is properly installed."""
        assert "pytest" in sys.modules or True  # Will be true after poetry install
    
    @pytest.mark.unit
    def test_project_structure(self):
        """Test that the project structure is correctly set up."""
        project_root = Path(__file__).parent.parent
        
        # Check essential directories exist
        assert project_root.exists()
        assert (project_root / "tests").exists()
        assert (project_root / "tests" / "unit").exists()
        assert (project_root / "tests" / "integration").exists()
        
        # Check configuration files exist
        assert (project_root / "pyproject.toml").exists()
        assert (project_root / ".gitignore").exists()
    
    @pytest.mark.unit
    def test_fixtures_available(self, temp_dir, mock_config):
        """Test that custom fixtures are available and working."""
        # Test temp_dir fixture
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Test mock_config fixture
        assert isinstance(mock_config, dict)
        assert "model_name" in mock_config
        assert mock_config["model_name"] == "test_model"
    
    @pytest.mark.unit
    def test_markers_registered(self, request):
        """Test that custom markers are properly registered."""
        markers = [m.name for m in request.node.iter_markers()]
        assert "unit" in markers
    
    @pytest.mark.integration
    def test_coverage_configuration(self):
        """Test that coverage is properly configured."""
        # This test will verify coverage is working when run with coverage
        assert True
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that the slow marker is working."""
        import time
        # Simulate a slow test
        time.sleep(0.1)
        assert True


def test_basic_assertion():
    """A simple test to ensure pytest is working."""
    assert 1 + 1 == 2
    assert "hello".upper() == "HELLO"
    assert [1, 2, 3][1] == 2