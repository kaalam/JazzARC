# Jazz (c) 2018-2023 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses

import numpy as np

from Bebop		import Bebop
from Block		import Block
from BopForward	import BopForward
from Code		import Code
from Context	import Context
from Core		import Core
from Function	import bop_function


def test_successful_end_to_end():
	ctx = Context('data')
	prb = ctx.get_problem_startswith('496994bd')

	code = Code((Bebop.bebop_get_question,
				 BopForward.bopforward_pic_all_as_pat,
				 BopForward.bopforward_pat_flip_up_down,
				 Bebop.bebop_get_question,
				 BopForward.bopforward_pic_all_as_pat,
				 BopForward.bopforward_2pat_merge_as_pic))

	for example in prb:
		q, a, _ = example.data

		core = Core(code)
		core.register['question'] = q

		*_, last_block = core

		assert core.all_right
		assert len(core.stack) == 1
		assert isinstance(last_block, Block)
		assert last_block.type == Block.type_picture

		assert np.array_equal(last_block.data, a.data)


class Failing():

	@bop_function(arg_types=[Block.type_core], ret_type=Block.type_nothing)
	def failing_emptystack(core):
		"""
		Returns an empty stack + Block.type_nothing.
		"""
		while core.stack:
			_ = core.stack.pop()


def test_all_the_corners():
	# 'pic1.shape != pic2.shape'

	pat1 = BopForward.bopforward_pic_all_as_pat.data(Block.new_picture(list_of_list=[[1, 0], [0, 1]]))
	pat2 = BopForward.bopforward_pic_all_as_pat.data(Block.new_picture(list_of_list=[[1, 0], [2, 1], [3, 2]]))

	ret = BopForward.bopforward_2pat_merge_as_pic.data(pat1, pat2)

	assert isinstance(ret, Block)
	assert ret.type == Block.type_error
	assert ret.data == 'pic1.shape != pic2.shape'

	# self.stack.append(block)
	# return self.hcf(ret.data)

	code = Code((pat1,
				 pat2,
				 BopForward.bopforward_2pat_merge_as_pic))

	core = Core(code)

	*_, last_block = core

	assert core.all_right is False
	assert core.error_msg == 'pic1.shape != pic2.shape'

	assert isinstance(last_block, Block)
	assert last_block.type == Block.type_error
	assert last_block.data == 'pic1.shape != pic2.shape'

	# StopIteration

	code = Code((pat1,
				 pat1,
				 BopForward.bopforward_2pat_merge_as_pic,
				 pat1,
				 pat2,
				 BopForward.bopforward_2pat_merge_as_pic,
				 pat2,
				 pat2,
				 BopForward.bopforward_2pat_merge_as_pic))

	core = Core(code)

	*_, last_block = core

	assert core.all_right is False
	assert core.error_msg == 'pic1.shape != pic2.shape'

	assert isinstance(last_block, Block)
	assert last_block.type == Block.type_error
	assert last_block.data == 'pic1.shape != pic2.shape'

	# 'Empty stack while unpacking arguments'

	code = Code((pat1,
				 BopForward.bopforward_2pat_merge_as_pic))

	core = Core(code)

	*_, last_block = core

	assert core.all_right is False
	assert core.error_msg == 'Empty stack while unpacking arguments'

	assert isinstance(last_block, Block)
	assert last_block.type == Block.type_error
	assert last_block.data == 'Empty stack while unpacking arguments'

	# 'Invalid Block type unpacked'

	core = Core(code, stack=[Block.new_picture(list_of_list=[[1]])])

	*_, last_block = core

	assert core.all_right is False
	assert core.error_msg == 'Invalid Block type unpacked'

	assert isinstance(last_block, Block)
	assert last_block.type == Block.type_error
	assert last_block.data == 'Invalid Block type unpacked'

	# 'Unexpected nothing return'

	@bop_function(arg_types=[Block.type_integer], ret_type=Block.type_int_pair)
	def fun_may_return_nothing(color):
		if color.data >= 2:
			return Block.new_int_pair((color.data - 1, color.data))

	code = Code((Block.new_integer(5),
				 fun_may_return_nothing))

	core = Core(code)

	*_, last_block = core

	assert core.all_right

	code = Code((Block.new_integer(1),
				 fun_may_return_nothing))

	core = Core(code)

	*_, last_block = core

	assert core.all_right is False
	assert core.error_msg == 'Unexpected nothing return'

	assert isinstance(last_block, Block)
	assert last_block.type == Block.type_error
	assert last_block.data == 'Unexpected nothing return'

	# 'Invalid Block type returned'

	@bop_function(arg_types=[Block.type_pattern], ret_type=Block.type_picture)
	def fun_returns_wrong_type(pat):
		return Block.new_integer(3)

	code = Code((pat1,
				 fun_returns_wrong_type))

	core = Core(code)

	*_, last_block = core

	assert core.all_right is False
	assert core.error_msg == 'Invalid Block type returned'

	assert isinstance(last_block, Block)
	assert last_block.type == Block.type_error
	assert last_block.data == 'Invalid Block type returned'

	# external function hcf()

	@bop_function(arg_types=[Block.type_core, Block.type_integer], ret_type=Block.type_integer)
	def fun_check_arg_and_hcf(core, color):
		if color.data < 1:
			core.hcf('Requires a color > 0')
		else:
			return Block.new_integer(2*color.data)

	code = Code((fun_check_arg_and_hcf, ))

	core = Core(code, stack=[Block.new_integer(4)])

	ret, = core

	assert core.all_right
	assert isinstance(ret, Block)
	assert ret.type == Block.type_integer
	assert ret.data == 8

	core = Core(code, stack=[Block.new_integer(-1)])

	ret, = core

	assert core.all_right is False
	assert core.error_msg == 'Requires a color > 0'

	assert isinstance(ret, Block)
	assert ret.type == Block.type_error
	assert ret.data == 'Requires a color > 0'

	# Empty stack after execution

	code = Code((Block.new_pictures(()), Failing.failing_emptystack))
	core = Core(code)

	*_, ret = core

	assert core.all_right is False
	assert core.error_msg == 'Empty stack after execution'

	assert isinstance(ret, Block)
	assert ret.type == Block.type_error
	assert ret.data == 'Empty stack after execution'
