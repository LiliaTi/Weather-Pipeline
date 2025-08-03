import sys
from pathlib import Path
import pytest

# Добавляем /app в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    pytest.main(["-v", "tests/"])