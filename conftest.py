"""Test bootstrap: the exercise test-suite calls ``self.assertIsNotNone``
without inheriting from ``unittest.TestCase``.  The test files must not be
modified, so provide the assertion here.
"""

import unittest

from tests.test_atmosphere import TestAtmosphereSystem

if not hasattr(TestAtmosphereSystem, "assertIsNotNone"):
    TestAtmosphereSystem.assertIsNotNone = unittest.TestCase.assertIsNotNone
