# Jazz (c) 2018-2020 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses

import numpy as np

from xgboost import XGBClassifier

from Block		import Block
from BopBack	import BopBack
from Code		import Code
from CodeGen	import CodeGen
from Context	import Context
from Example	import Example
from Field		import Field
from Search		import Search
from Source		import Source
from SearchConf	import LENGTH_CODE_EVAL


def test_run_all():
	context	= Context('data')

	Search.build_code_base('./code_base/basis_code.jcb', context)

	cg = CodeGen('./code_base/basis_code.jcb', None, Example)

	cg.build_reward_training_data('./code_base/basis_code.evd', context)

	cg = CodeGen(None, './code_base/basis_code.evd', Example)

	assert isinstance(cg.xgb, XGBClassifier)

	with open('./code_base/basis_code.evd', 'r') as f:
		txt = f.read().splitlines()

	lol = []

	for s in txt:
		if s[0] != '.':
			lol.append([float(x) for x in s.split(', ')])

	test = np.array(lol)

	pred = cg.predict_rewards(test[:, range(1, LENGTH_CODE_EVAL + 1)])

	assert sum(1*(pred > 0.5) == test[:, 0])/test.shape[0] > 0.8


def assert_error(source_tuple, error_msg):
	field = Field(Example)

	code = field.compile(Source(source_tuple))

	assert isinstance(code, Code)

	ret = field.run(code)

	assert isinstance(ret, Block)
	assert ret.type == Block.type_error
	assert ret.data == error_msg


def test_error_conditions():
	assert_error(('swap_top2',), 'swap_top2() with less than two')
	assert_error(('[[1, 2]]', '[[3]]', '2pic_maximum'), 'pic1.shape != pic2.shape')
	assert_error(('[[1, 2]]', 'pic_all_as_pat', '[[3]]', 'pic_all_as_pat', '2pat_merge'), 'pic1.shape != pic2.shape')
	assert_error(('[[1, 2]]', '[[3]]', '2pic_and_masks_to_1'), 'pic1.shape != pic2.shape')
	assert_error(('[[1, 2]]', '[[3]]', '2pic_xor_masks_to_1'), 'pic1.shape != pic2.shape')
	assert_error(('(4, 2)', '[[1, 2], [1, 2], [1, 2]]', 'pic_intp_fork_on_shape_as_pics'), 'Pic height not a multiple of output')
	assert_error(('(3, 3)', '[[1, 2], [1, 2], [1, 2]]', 'pic_intp_fork_on_shape_as_pics'), 'Pic width not a multiple of output')
	assert_error(('(5, 7)', '[[1, 2], [1, 2], [1, 2]]', 'pic_intp_fork_on_shape_as_pics'), 'Neither height nor width match output')

	assert_error(('(2, 2)', '[[1, 0, 0, 1], [0, 1, 1, 0]]', 'pic_intp_fork_on_shape_as_pics', 'pics_filter_unique_pattern_as_pic'),
				  'More than one unique pictures')
	assert_error(('(2, 2)', '[[1, 0, 1, 0], [0, 1, 0, 1]]', 'pic_intp_fork_on_shape_as_pics', 'pics_filter_unique_pattern_as_pic'),
				  'No unique pictures')

	assert_error(('(2, 2)', '[[1, 0, 0, 1], [0, 1, 1, 0]]', 'pic_intp_fork_on_shape_as_pics', 'pics_filter_unique_picture_as_pic'),
				  'More than one unique pictures')
	assert_error(('(2, 2)', '[[1, 0, 1, 0], [0, 1, 0, 1]]', 'pic_intp_fork_on_shape_as_pics', 'pics_filter_unique_picture_as_pic'),
				  'No unique pictures')

	assert_error(('swap_top3',), 'swap_top3() with less than three')
	assert_error(('(2, 2)', '(3, 3)', '[[1, 0, 1], [0, 1, 0]]', 'pic_2intp_crop'), 'Image too small for required crop')
	assert_error(('(3,)', '[[1, 0, 1], [0, 1, 0]]', 'pic_int_zoom_out'), 'Wrong shapes for scale')

	assert_error(('get_a',), 'get_a() empty register')
	assert_error(('get_b',), 'get_b() empty register')
	assert_error(('get_c',), 'get_c() empty register')
	assert_error(('get_d',), 'get_d() empty register')
	assert_error(('get_e',), 'get_e() empty register')

	assert_error(('get_answer',),	'get_answer() empty register')
	assert_error(('get_question',),	'get_question() empty register')

	assert_error(('sto_a',), 'sto_a() empty stack')
	assert_error(('sto_b',), 'sto_b() empty stack')
	assert_error(('sto_c',), 'sto_c() empty stack')
	assert_error(('sto_d',), 'sto_d() empty stack')
	assert_error(('sto_e',), 'sto_e() empty stack')

	assert_error(('[[1,2,3,1,2,3,1,2],[2,1,3,2,1,3,2,1]]', 'pic_fork_on_auto_grid_as_pics', 'pics_as_pic'),
				 'pics_as_pic() tuple of one picture expected')
	assert_error(('[[1,2,3,1,2,3,1,2],[2,1,3,2,1,3,2,1]]', 'pic_fork_on_auto_grid_as_pics', 'pics_as_2pic'),
				 'pics_as_2pic() tuple of two pictures expected')

	assert_error(('[[1,2,3,1,2],[2,1,3,2,1]]', 'pic_fork_on_auto_grid_as_pics', 'pics_as_pic'),
				 'pics_as_pic() tuple of one picture expected')
	assert_error(('[[1,2,3,1,2],[2,1,3,2,1]]', 'pic_fork_on_auto_grid_as_pics', 'pics_as_3pic'),
				 'pics_as_3pic() tuple of three pictures expected')

	assert_error(('[[1,2],[2,1]]', 'pic_fork_on_auto_grid_as_pics', 'pics_as_2pic'),
				 'pics_as_2pic() tuple of two pictures expected')
	assert_error(('[[1,2],[2,1]]', 'pic_fork_on_auto_grid_as_pics', 'pics_as_3pic'),
				 'pics_as_3pic() tuple of three pictures expected')

	assert_error(('[[1,2],[2,1]]', 'pic_all_as_pat', '[[1,2,3],[3,2,1]]', 'pic_all_as_pat', '2pat_merge_if_disjoint_as_pic'),
				  'pic1.shape != pic2.shape')

	assert_error(('[[1,2,3,2,1],[2,1,3,1,2]]', 'pic_fork_on_auto_grid_as_pics', 'pics_filter_single_color'), 'Filter not found')
	assert_error(('[[1,2,3,2,1],[2,1,3,1,2]]', 'pic_fork_on_auto_grid_as_pics', 'pics_filter_v_symmetric'), 'Filter not found')

	assert_error(('[[1,1,1,1]]', '[[1,2,5,3,2,1],[2,1,5,1,0,3]]', 'pic_fork_on_auto_grid_as_pics', 'pics_pic_multiply_as_pic'),
				  'All bricks must have the same shape')

	assert_error(('[[1,1,1]]', '[[1,2,5,3,1],[2,1,5,1,3]]', 'pic_fork_on_auto_grid_as_pics', 'pics_pic_multiply_as_pic'),
				  'Output cells not a multiple of bricks')

	assert_error(('[[1,2,3,1,2],[2,1,3,2,1]]', '[[4,5,6]]', '2pic_cbind'), 'pic1.shape and pic2.shape not same height')
	assert_error(('[[1,2,3,1,2],[2,1,3,2,1]]', '[[4,5,6]]', '2pic_rbind'), 'pic1.shape and pic2.shape not same width')
	assert_error(('[[1,2,3,1,2],[2,1,3,2,1]]', 'pic_two_col_reverse'), 'Only two colors expected')
	assert_error(('[[1,2,1,2],[2,1,2,1]]', 'pic_v_axis'), 'pic_v_axis() expects odd width')
	assert_error(('[1,2]', 'vec_as_int'), 'Expects vector of length 1')

	assert_error(('(2,3)', '[[1,1,2,2],[1,1,2,2],[3,3,4,4],[3,3,4,4]]', 'pic_intp_zoom_fit'), 'Not an integer scale for zoom_out')
	assert_error(('(4,5)', '[[1,2],[3,4]]', 'pic_intp_zoom_fit'), 'Not an integer scale for zoom_in')
	assert_error(('(2,4)', '[[1,2,3],[1,2,3]]', 'pic_intp_zoom_fit'), 'X scale != Y scale')

	assert_error(('[[0,0],[0,0]]', 'pic_fork_by_color_as_pics', 'pics_maximum_as_pic'), 'Pics cannot be empty')
	assert_error(('[[1,2,3,8],[2,2,2,2],[4,2,5,6],[7,2,6,5]]', 'pic_fork_on_auto_grid_as_pics', 'pics_maximum_as_pic'),
				  'All pics must have the same shape')


def assert_equal(source_tuple, add_compare=True, val=1):
	field = Field(Example)

	code = field.compile(Source(source_tuple))

	assert isinstance(code, Code)

	example = None
	if add_compare:
		code = Code(code.data + (BopBack.bopback_tests_verify_answer,))

		answer = code.data[0]

		example = Example([4, 0, 4], answer.data.tolist(), False)

	ret = field.run(code, example)

	assert isinstance(ret, Block)
	assert ret.type == Block.type_integer
	assert isinstance(ret.data, int)

	assert ret.data == val


def test_assert_equal():
	assert_equal(('[[1,2,3,0],[0,4,5,0],[0,0,0,0]]',
				  '(1,0,0,1)',
				  '[[0,0,0,0],[0,1,2,3],[0,0,4,5]]',
				  'pic_all_as_pat',
				  'pat_nesw_drag_all',
				  'pat_as_pic'))

	assert_equal(('[[0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,4,0,0]]',
				  '(1,4,3,2)',
				  '[[0,0,0,0,0,0], [0,4,3,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0]]',
				  'pic_all_as_pat',
				  'pat_nesw_drag_all',
				  'pat_as_pic'))

	assert_equal(('[[0,0,0,0,0],[0,1,2,0,0],[0,3,4,0,0],[0,0,0,0,0],[0,0,0,0,0]]',
				  '(1,2,2,1)',
				  '[[1,2],[3,4]]',
				  'pic_nesw_extend'))

	assert_equal(('[[0,0,0,0,0,0,0,0],[0,0,0,0,1,2,0,0],[0,0,0,0,3,4,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]',
				  '(1,2,3,4)',
				  '[[1,2],[3,4]]',
				  'pic_nesw_extend'))

	assert_equal(('[[2,0,1,0,4],[0,4,0,3,0],[2,0,5,0,4]]',
				  '[[0,2,4],[0,0,4]]',
				  '[[1,0,1,0,1],[0,3,0,3,0],[5,0,5,0,5]]',
				  '2pic_recolor_any_rtl'))

	assert_equal(('[[4,0,4]]',
				  '[[1,2,3],[4,5,6],[7,7,7],[1,2,3],[4,5,6],[7,7,7],[4,0,4]]',
				  'pic_fork_on_auto_grid_as_pics',
				  'pics_filter_unique_picture_as_pic'))

	assert_equal(('[[4],[0],[4]]',
				  '[[4,2,3,5,2,3,5],[0,2,5,3,2,5,3],[4,2,3,5,2,3,5]]',
				  'pic_fork_on_auto_grid_as_pics',
				  'pics_filter_unique_picture_as_pic'))

	assert_equal(('[[1,3], [3,1]]',
				  '[[1,3], [3,1]]',
				  'pic_fork_on_auto_grid_as_pics',
				  'pics_filter_unique_picture_as_pic'))

	assert_equal(('[[0],[0],[0]]',
				  '[[1,2,3],[4,5,6],[7,7,7],[1,2,3],[4,5,6],[7,7,7],[4,0,4]]',
				  'pic_shape_on_auto_grid'))

	assert_equal(('[[0,0,0]]',
				  '[[4,2,3,5,2,3,5],[0,2,5,3,2,5,3],[4,2,3,5,2,3,5]]',
				  'pic_shape_on_auto_grid'))

	assert_equal(('[[0]]',
				  '[[1,3], [3,1]]',
				  'pic_shape_on_auto_grid'))

	assert_equal(('[[0,0,0],[0,0,0]]',
				  'pic_base_height_as_int'), add_compare=False, val=0)

	assert_equal(('[[0,0,0], [0,0,0], [0,0,0], [0,0,0], [3,0,0], [3,3,3], [0,0,0]]',
				  '[[0,3,0], [0,3,0], [0,3,0], [0,3,0], [0,3,0], [3,3,3], [0,0,0]]',
				  'sto_a',
				  'get_a',
				  'pic_base_height_as_int',
				  'swap_top2',
				  'pic_int_slide_rows_west'))

	assert_equal(('(1,)', 'sto_a', '(2,)', 'sto_b', '(3,)', 'sto_c', '(4,)', 'sto_d', '(5,)', 'sto_e',
				  'get_a'), add_compare=False, val=1)
	assert_equal(('(1,)', 'sto_a', '(2,)', 'sto_b', '(3,)', 'sto_c', '(4,)', 'sto_d', '(5,)', 'sto_e',
				  'get_b'), add_compare=False, val=2)
	assert_equal(('(1,)', 'sto_a', '(2,)', 'sto_b', '(3,)', 'sto_c', '(4,)', 'sto_d', '(5,)', 'sto_e',
				  'get_c'), add_compare=False, val=3)
	assert_equal(('(1,)', 'sto_a', '(2,)', 'sto_b', '(3,)', 'sto_c', '(4,)', 'sto_d', '(5,)', 'sto_e',
				  'get_d'), add_compare=False, val=4)
	assert_equal(('(1,)', 'sto_a', '(2,)', 'sto_b', '(3,)', 'sto_c', '(4,)', 'sto_d', '(5,)', 'sto_e',
				  'get_e'), add_compare=False, val=5)

	assert_equal(('[[1,2],[2,1]]', '[[1,2,3,1,2,3,1,2],[2,1,3,2,1,3,2,1]]', 'pic_fork_on_auto_grid_as_pics', 'pics_as_3pic'))
	assert_equal(('[[1,2],[2,1]]', '[[1,2,3,1,2],[2,1,3,2,1]]', 'pic_fork_on_auto_grid_as_pics', 'pics_as_2pic'))
	assert_equal(('[[1,2],[2,1]]', '[[1,2],[2,1]]', 'pic_fork_on_auto_grid_as_pics', 'pics_as_pic'))

	assert_equal(('[[1,2],[2,1]]', '(1,)'), val=0)										# tests_verify_answer is added automatically
	assert_equal(('[[1,2],[2,1]]', 'tests_verify_answer'), add_compare=False, val=0)
	assert_equal(('tests_verify_answer',), add_compare=False, val=0)

	assert_equal(('[[1,2],[3,4]]', '(2,2)', '[[1,1,2,2],[1,1,2,2],[3,3,4,4],[3,3,4,4]]', 'pic_intp_zoom_fit'))
	assert_equal(('[[1,1,2,2],[1,1,2,2],[3,3,4,4],[3,3,4,4]]', '(4,4)', '[[1,2],[3,4]]', 'pic_intp_zoom_fit'))
	assert_equal(('[[1,2,3],[1,2,3]]', '(2,3)', '[[1,2,3],[1,2,3]]', 'pic_intp_zoom_fit'))

	assert_equal(('[[0,1,2],[0,5,4]]', '(0,1,0,0)', '[[1,2,0],[5,0,4]]', 'pic_nesw_gravity'))
	assert_equal(('[[1,2,0],[5,4,0]]', '(0,0,0,1)', '[[1,0,2],[5,0,4]]', 'pic_nesw_gravity'))
