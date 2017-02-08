# -*- coding: utf-8 -*-
u"""
unittest code

author Atsushi Sakai
"""
import unittest
import pyecosqp.pyecosqp


class Test(unittest.TestCase):

    def test_1(self):
        pyecosqp.pyecosqp.test1()

    def test_2(self):
        pyecosqp.pyecosqp.test2()

    def test_3(self):
        pyecosqp.pyecosqp.test2()


if __name__ == '__main__':
    unittest.main()
