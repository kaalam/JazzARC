# Jazz (c) 2018-2024 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses

from Block	 import Block


class bop_function():
	"""
	This decorator makes a method of an abstract class a Bebop function. This is a decorator with arguments.
	"""

	def __init__(self, arg_types, ret_type):
		"""
		See https://python-3-patterns-idioms-test.readthedocs.io/en/latest/PythonDecorators.html
		If there are decorator arguments, the function to be decorated is not passed to the constructor!
		"""
		self.arg_types = arg_types
		self.ret_type  = ret_type


	def __call__(self, fun):
		"""
		See https://python-3-patterns-idioms-test.readthedocs.io/en/latest/PythonDecorators.html
		If there are decorator arguments, __call__() is only called once, as part of the decoration process! You can only give
		it a single argument, which is the function object.
		"""
		return Function(fun, self.arg_types, self.ret_type)


class Function(Block):
	"""
	All executable functions and code descend from this. It is a wrapper over Block that forces the object to have a list of
	arguments types and a result type. All types in both arg_types[] and ret_type must be in Block.type_*. arg_types is always
	a list, even if it only contains one element.
	"""

	def __init__(self, call, arg_types, ret_type):
		self.data = call
		self.type = Block.type_function
		self.arg_types = arg_types
		self.ret_type  = ret_type
