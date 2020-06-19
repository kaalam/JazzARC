# Jazz (c) 2018-2020 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses

import pytest

import numpy as np

from Block		import Block
from Code		import Code
from CodeEval	import CodeEval
from Context	import Context
from Example	import Example
from Function	import bop_function
from Source		import Source
from SearchConf	import EVAL_FULL_MATCH
from SearchConf	import EVAL_WRONG_SHAPE
from SearchConf	import IDX_PIC_REACH_MIN
from SearchConf	import IDX_PIC_BETTER_MIN
from SearchConf	import IDX_PIC_WORSE_MIN
from SearchConf	import IDX_PAT_REACH_MIN
from SearchConf	import IDX_PAT_BETTER_MIN
from SearchConf	import IDX_PAT_WORSE_MIN
from SearchConf	import IDX_PIC_REACH_MEAN
from SearchConf	import IDX_PIC_BETTER_MEAN
from SearchConf	import IDX_PIC_WORSE_MEAN
from SearchConf	import IDX_PAT_REACH_MEAN
from SearchConf	import IDX_PAT_BETTER_MEAN
from SearchConf	import IDX_PAT_WORSE_MEAN
from SearchConf	import IDX_PIC_REACH_MAX
from SearchConf	import IDX_PIC_BETTER_MAX
from SearchConf	import IDX_PIC_WORSE_MAX
from SearchConf	import IDX_PAT_REACH_MAX
from SearchConf	import IDX_PAT_WORSE_MAX


class Failing():

	@bop_function(arg_types=[Block.type_integer], ret_type=Block.type_integer)
	def failing_zero_division(dd):
		"""
		raises a ZeroDivisionError when dd is 0.
		"""
		return Block.new_integer(1/dd.data)


def test_MulticoreErrors():
	eval = CodeEval(Example)

	eval.demo_q	  = [Block.new_picture(list_of_list=[[1, 2, 3]])]
	eval.demo_a	  = [Block.new_picture(list_of_list=[[4, 0, 4]])]
	eval.question = []

	eval.multicore_clear()

	ret = eval.multicore_run_all(Code((Block.new_integer(1), Failing.failing_zero_division,
									   Block.new_integer(0), Failing.failing_zero_division)))

	assert ret.type == Block.type_error
	assert ret.data == 'Try/catch caught an exception'

	eval.multicore_clear()

	ret = eval.multicore_run_all(eval.compile(Source(('get_question', 'pic_two_col_reverse'))))

	assert ret.type == Block.type_error
	assert ret.data == 'Only two colors expected'

	eval.multicore_clear()

	ret = eval.multicore_run_all(eval.compile(Source(('get_question', 'pic_fork_on_v_axis_as_pics', 'pics_as_2pic'))))

	assert ret.type == Block.type_no_error

	for pl in eval.multicore_state['pic_lists']:
		assert len(pl) == 1

	ret = eval.multicore_run_all(eval.compile(Source(('(2, 4)',))))

	assert ret.type == Block.type_error
	assert ret.data == 'Code item does not return a picture'

	ret = eval.multicore_run_all(eval.compile(Source(('(2, 4)',))), ignore_ret_type=True)

	assert ret.type == Block.type_no_error


def test_Multicore():
	# '1b2d62fb.json' : Source(('get_question',
	#   						'pic_fork_on_v_axis_as_pics',
  	# 							'pics_as_2pic',
  	# 							'2pic_maximum',
	# 							'(0, 8)',
	# 							'swap_top2',
	# 							'pic_intp_swap_colors',
	# 							'(9, 0)',
	# 							'swap_top2',
	# 							'pic_intp_swap_colors')),
	ctx = Context('data')

	prob = ctx.get_problem_startswith('1b2d62fb')

	eval = CodeEval(Example)

	eval.demo_q	  = []
	eval.demo_a	  = []
	eval.question = []

	for example in prob:
		q, a, is_test = example.data
		if is_test:
			eval.question.append(q)
		else:
			eval.demo_q.append(q)
			eval.demo_a.append(a)

	c11 = eval.compile(Source(('get_question', 'pic_fork_on_v_axis_as_pics', 'pics_as_2pic', '2pic_maximum')))
	c12 = eval.compile(Source(('get_question', 'pic_fork_on_h_axis_as_pics', 'pics_as_2pic', '2pic_maximum')))
	c13 = eval.compile(Source(('get_question', 'pic_fork_on_v_axis_as_pics', 'pics_as_2pic', '2pic_xor_masks_to_1')))

	c21 = eval.compile(Source(('(0, 8)', 'swap_top2', 'pic_intp_swap_colors')))
	c22 = eval.compile(Source(('(5, 4)', 'swap_top2', 'pic_intp_swap_colors')))
	c23 = eval.compile(Source(('(5, 4)', 'swap_top2', 'pic_intp_select_columns')))

	c31 = eval.compile(Source(('(9, 0)', 'swap_top2', 'pic_intp_swap_colors')))
	c32 = eval.compile(Source(('(5, 4)', 'swap_top2', 'pic_intp_recolor')))
	c33 = eval.compile(Source(('pic_transpose',)))

	assert c11.type == Block.type_no_error and c12.type == Block.type_no_error and c13.type == Block.type_no_error
	assert c21.type == Block.type_no_error and c22.type == Block.type_no_error and c23.type == Block.type_no_error
	assert c31.type == Block.type_no_error and c32.type == Block.type_no_error and c33.type == Block.type_no_error

	eval.multicore_clear()

	ret = eval.multicore_run_all(c11)

	assert ret.type == Block.type_no_error

	for pl in eval.multicore_state['pic_lists']:
		assert len(pl) == 1

	ret = eval.multicore_run_all(c21)

	assert ret.type == Block.type_no_error

	for pl in eval.multicore_state['pic_lists']:
		assert len(pl) == 2

	ret = eval.multicore_run_all(c31)

	assert ret.type == Block.type_no_error

	for pic_list, example in zip(eval.multicore_state['pic_lists'], prob):
		assert len(pic_list) == 3

		_, answer, _ = example.data

		assert np.array_equal(pic_list[2].data, answer.data)

	eval.multicore_clear()

	root = eval.multicore_copy_state(eval.multicore_state)

	ret = eval.multicore_run_all(c13)
	assert ret.type == Block.type_no_error
	ret = eval.multicore_run_all(c21)
	assert ret.type == Block.type_no_error

	for pl in eval.multicore_state['pic_lists']:
		assert len(pl) == 2

	eval.multicore_set_state(root)

	for pl in eval.multicore_state['pic_lists']:
		assert len(pl) == 0

	ret = eval.multicore_run_all(c12)
	assert ret.type == Block.type_no_error
	ret = eval.multicore_run_all(c23)
	assert ret.type == Block.type_no_error

	for pl in eval.multicore_state['pic_lists']:
		assert len(pl) == 2

	eval.multicore_set_state(root)

	ret = eval.multicore_run_all(c11)

	for pl in eval.multicore_state['pic_lists']:
		assert len(pl) == 1

	lev_1 = eval.multicore_copy_state(eval.multicore_state)

	ret = eval.multicore_run_all(c23)
	assert ret.type == Block.type_no_error
	ret = eval.multicore_run_all(c33)
	assert ret.type == Block.type_no_error

	eval.multicore_set_state(lev_1)

	ret = eval.multicore_run_all(c22)
	assert ret.type == Block.type_no_error
	ret = eval.multicore_run_all(c31)
	assert ret.type == Block.type_no_error

	eval.multicore_set_state(lev_1)

	ret = eval.multicore_run_all(c21)

	lev_2 = eval.multicore_copy_state(eval.multicore_state)

	ret = eval.multicore_run_all(c33)
	assert ret.type == Block.type_no_error

	eval.multicore_set_state(lev_2)

	ret = eval.multicore_run_all(c32)
	assert ret.type == Block.type_no_error

	eval.multicore_set_state(lev_2)

	ret = eval.multicore_run_all(c31)

	assert ret.type == Block.type_no_error

	pic_lists = eval.multicore_state['pic_lists']

	for pic_list, example in zip(pic_lists, prob):
		assert len(pic_list) == 3

		_, answer, _ = example.data

		assert np.array_equal(pic_list[2].data, answer.data)


def test_CodeEval():
	# '1b2d62fb.json' : Source(('get_question',
	#   						'pic_fork_on_v_axis_as_pics',
  	# 							'pics_as_2pic',
  	# 							'2pic_maximum',
	# 							'(0, 8)',
	# 							'swap_top2',
	# 							'pic_intp_swap_colors',
	# 							'(9, 0)',
	# 							'swap_top2',
	# 							'pic_intp_swap_colors')),
	ctx = Context('data')

	prob = ctx.get_problem_startswith('1b2d62fb')

	eval = CodeEval(Example)

	with pytest.raises(TypeError):			# self.multicore_state is None
		eval.eval_code()

	eval.demo_q	  = []
	eval.demo_a	  = []
	eval.question = []

	eval.multicore_clear()					# Builds a state without cores

	with pytest.raises(LookupError):		# self.multicore_state is None
		eval.eval_code()

	for example in prob:
		q, a, is_test = example.data
		if is_test:
			eval.question.append(q)
		else:
			eval.demo_q.append(q)
			eval.demo_a.append(a)

	eval.multicore_clear()

	with pytest.raises(LookupError):		# Runs an eval without a picture
		eval.eval_code()

	c1 = eval.compile(Source(('get_question', 'pic_fork_on_v_axis_as_pics', 'pics_as_2pic', '2pic_maximum')))
	c2 = eval.compile(Source(('(0, 8)', 'swap_top2', 'pic_intp_swap_colors')))
	c3 = eval.compile(Source(('(9, 0)', 'swap_top2', 'pic_intp_swap_colors')))

	ret = eval.multicore_run_all(c1)
	assert ret.type == Block.type_no_error

	vec = eval.eval_code()

	assert vec[IDX_PIC_REACH_MIN] == 0 and vec[IDX_PIC_REACH_MAX] == 0
	assert vec[IDX_PAT_REACH_MIN] == 0 and vec[IDX_PAT_REACH_MAX] == 0
	assert vec[IDX_PIC_BETTER_MEAN] == EVAL_WRONG_SHAPE and vec[IDX_PIC_WORSE_MEAN] == EVAL_WRONG_SHAPE

	ret = eval.multicore_run_all(c2)
	assert ret.type == Block.type_no_error

	vec = eval.eval_code()

	assert 0 < vec[IDX_PIC_REACH_MIN] and vec[IDX_PIC_REACH_MIN] < vec[IDX_PIC_REACH_MEAN]
	assert vec[IDX_PIC_REACH_MEAN] < vec[IDX_PIC_REACH_MAX] and vec[IDX_PIC_REACH_MAX] < 1

	assert vec[IDX_PIC_REACH_MIN] == vec[IDX_PAT_REACH_MIN] and vec[IDX_PIC_REACH_MEAN] == vec[IDX_PAT_REACH_MEAN]

	assert vec[IDX_PIC_WORSE_MAX] == 0 and vec[IDX_PAT_WORSE_MAX] == 0

	assert 0 < vec[IDX_PIC_BETTER_MIN] and vec[IDX_PIC_BETTER_MIN] < vec[IDX_PIC_BETTER_MEAN]
	assert vec[IDX_PIC_BETTER_MEAN] < vec[IDX_PIC_BETTER_MAX] and vec[IDX_PIC_BETTER_MAX] < 1

	assert vec[IDX_PIC_BETTER_MIN] == vec[IDX_PAT_BETTER_MIN] and vec[IDX_PIC_BETTER_MEAN] == vec[IDX_PAT_BETTER_MEAN]

	ret = eval.multicore_run_all(c3)
	assert ret.type == Block.type_no_error

	vec = eval.eval_code()

	assert vec[IDX_PIC_REACH_MIN] == EVAL_FULL_MATCH
	assert vec[IDX_PAT_REACH_MIN] == EVAL_FULL_MATCH

	assert vec[IDX_PIC_WORSE_MIN]  == 0 and vec[IDX_PAT_WORSE_MIN]  == 0
	assert vec[IDX_PIC_WORSE_MEAN] == 0 and vec[IDX_PAT_WORSE_MEAN] == 0
	assert vec[IDX_PIC_WORSE_MAX]  == 0 and vec[IDX_PAT_WORSE_MAX]  == 0

	assert 0 < vec[IDX_PIC_BETTER_MIN] and vec[IDX_PIC_BETTER_MIN] < vec[IDX_PIC_BETTER_MEAN]
	assert vec[IDX_PIC_BETTER_MEAN] < vec[IDX_PIC_BETTER_MAX] and vec[IDX_PIC_BETTER_MAX] < 1

	eval.multicore_clear()

	c1 = eval.compile(Source(('get_question', 'pic_transpose')))

	ret = eval.multicore_run_all(c1)
	assert ret.type == Block.type_no_error

	vec = eval.eval_code()

	for vv in vec.tolist():
		assert vv == EVAL_WRONG_SHAPE
