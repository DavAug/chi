#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import unittest

import erlotinib as erlo


class TestDummyFunction(unittest.TestCase):

    def test_output(self):
        self.assertEqual(erlo.dummy_function(), 42)


if __name__ == '__main__':
    unittest.main()
