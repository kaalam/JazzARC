# Jazz (c) 2018-2026 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses

from collections import deque

# import numpy as np

from Block import Block


class Core(Block):
	"""
	A Core is the unit of execution. It has a stack, initialized on construction and a Code object to be executed.
	A Core is an iterator executing one function call per iteration.
	Each iteration returns the last block pushed to the stack (without poping it).

	A complete execution of a Code snippet looks like:

	`core = Core(code, state_of_the_stack)`
	`*_, last_block = core`
	`if core.all_right:`
	`	do_something_with(last_block)`

	The stack argument (is empty by default) is a list of Block objects pushed to the stack (LIFO) before execution.
	"""

	def __init__(self, code, stack=None, register=None):
		self.data = code.data

		if stack is None:
			self.stack = deque()
		elif isinstance(stack, list):
			self.stack = deque(stack)
		else:
			self.stack = stack

		# self.question = None
		if register is None:
			self.register = {'a' : None, 'b' : None, 'c' : None, 'd' : None, 'e' : None, 'question' : None, 'answer' : None}
		else:
			self.register = register
			# if self.register['question'] is not None:
			# 	self.question = self.register['question'].data.copy()

		self.all_right = True
		self.error_msg = None
		self.ret	   = None


	def __iter__(self):
		"""
		This is the iterator. It is a Python Generator using yield() to return the block on top of the stack after each iteration.
		"""
		for block in self.data:
			if self.all_right:
				self.execute(block)
				yield(self.ret)
			else:
				return			# This is how you stop a generator, not using `raise StopIteration`


	def execute(self, block):
		"""
		This method executes Bebop code (at all levels). The method Field.run() is just a wrapper over this, initializing the stack
		from either the ori or dst of the Example bond.
		"""
		if block.type == Block.type_function:
			args = ()
			for typ in block.arg_types:
				if typ == Block.type_core:
					arg = self
				else:
					if not self.stack:
						return self.hcf('Empty stack while unpacking arguments')
					arg = self.stack.pop()
					if arg.type != typ:
						return self.hcf('Invalid Block type unpacked')

				args = args + (arg,)

			ret = block.data(*args)

			# if self.question is not None:
			# 	if not np.array_equal(self.question, self.register['question'].data):
			# 		self.hcf('What??')

			if ret is None:
				if block.ret_type != Block.type_nothing:
					return self.hcf('Unexpected nothing return')
			else:
				if ret.type == block.ret_type:
					self.stack.append(ret)
				else:
					if ret.type == Block.type_error:
						return self.hcf(ret.data)
					else:
						return self.hcf('Invalid Block type returned')
		else:
			self.stack.append(block)

		if not self.stack:
			return self.hcf('Empty stack after execution')		# Was possible when core_pics_explode_and_push() exploded a (). Now,
																# any opcode generating this should be treated as a bug.

		self.ret = self.stack[-1]


	def hcf(self, error_msg):
		"""
		Halt and Catch Fire

		This breaks the Core execution forever and will be called on any error. After hcf() is called only post-mortem
		analysis can be done.

		When called externally, it is the way to force the Core iterator to stop. Core functions may not return anything.
		Therefore, they cannot return a Block.new_error() but, since they have the core passed as an argument, they core.hcf().
		"""

		if self.all_right:										# In case of nested errors, set to the first one.
			self.all_right = False
			self.error_msg = error_msg
			self.stack.append(Block.new_error(error_msg))		# Redundant, but forces iterator returned block to be of type_error.
			self.ret = self.stack[-1]
