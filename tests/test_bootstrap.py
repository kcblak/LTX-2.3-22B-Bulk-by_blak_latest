import unittest
from collections import namedtuple
from unittest.mock import patch

from bootstrap import check_disk_space
from core import BootstrapError


class BootstrapTests(unittest.TestCase):
    def test_check_disk_space_raises_on_low_free_space(self):
        usage = namedtuple("usage", ["total", "used", "free"])(
            total=20 * 1024**3,
            used=15 * 1024**3,
            free=5 * 1024**3,
        )
        with patch("bootstrap.shutil.disk_usage", return_value=usage):
            with self.assertRaises(BootstrapError):
                check_disk_space()


if __name__ == "__main__":
    unittest.main()
