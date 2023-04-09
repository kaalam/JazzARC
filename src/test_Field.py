# Jazz (c) 2018-2023 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses

import pytest

from Answer		import Answer
from Block		import Block
from BopForward	import BopForward
from Code		import Code
from Context	import Context
from Example	import Example
from Field		import Field
from Question	import Question
from Source		import Source


def test_successful_end_to_end():
	f = Field(Example)

	assert f.from_kind == Question
	assert f.to_kind   == Answer

	assert 'get_question'		 in f.opcode
	assert 'tests_verify_answer' in f.opcode
	assert '2pat_merge_as_pic'	 in f.opcode

	source = Source(('get_question',
					 'pic_all_as_pat',
					 'pat_flip_up_down',
					 'get_question',
					 'pic_all_as_pat',
					 '2pat_merge_as_pic',
					 'tests_verify_answer'))

	code = f.compile(source)

	assert isinstance(code, Code)
	assert len(code.data) == len(source.data)

	ctx = Context('data')
	prb = ctx.get_problem_startswith('496994bd')

	for example in prb:
		ret = f.run(code, example, peek_answer=True)

		assert isinstance(ret, Block)
		assert ret.type == Block.type_integer
		assert isinstance(ret.data, int)

		assert ret.data == 1


def test_decompile():
	field = Field(Example)

	source = ('[2, 4, 6, 8]', '(0, 1, 1, 0)', '(1, 2)', '(2,)', '[[1, 1], [1, 0]]',
			  'pic_int_zoom_in', 'pic_intp_select_columns', 'pic_nesw_extend', 'pic_vec_recolor_each')

	code = field.compile(Source(source))

	assert code.type == Block.type_no_error

	new_source = field.decompile(code)

	assert field.rex_vector.match(new_source[0].strip())
	assert field.rex_tuple.match(new_source[1].strip())
	assert field.rex_tuple.match(new_source[2].strip())
	assert new_source[3].strip() == '2'
	assert field.rex_picture.match(new_source[4].strip())

	new_source = field.decompile(code, pretty=False)

	assert new_source == source

	new_source = field.decompile(BopForward.bopforward_2pic_recolor_any_rtl)

	assert len(new_source) == 1

	new_source = field.decompile((Block.new_picture(list_of_list=[[4, 0, 4]]), BopForward.bopforward_pic_all_as_pat))

	assert len(new_source) == 2

	with pytest.raises(AttributeError):
		field.decompile((Block.new_picture(list_of_list=[[4, 0, 4]]), 'aa'))



def test_all_the_corners():
	f = Field(Example)

	# picture (good)
	source = Source(('[[1, 0, 0], [0, 2, 2], [0, 0, 0]]',
					 'pic_all_as_pat'))

	code = f.compile(source)

	assert isinstance(code, Code)
	assert len(code.data) == len(source.data)

	ret = f.run(code)

	assert isinstance(ret, Block)
	assert ret.type == Block.type_pattern

	# picture (bad)
	source = Source(('[[1, 0, 0]], [0, 2, 2], [0, 0, 0]]',
					 'pic_all_as_pat'))

	code = f.compile(source)

	assert isinstance(code, Block)
	assert code.type == Block.type_error
	assert code.data.startswith('Malformed picture')

	# tuple (good)
	source = Source(('(9,)',
					 '(10,11)',
					 'swap_top2'))

	code = f.compile(source)

	assert isinstance(code, Code)
	assert len(code.data) == len(source.data)

	ret = f.run(code)

	assert isinstance(ret, Block)
	assert ret.type == Block.type_integer
	assert isinstance(ret.data, int)
	assert ret.data == 9

	# tuple (good)
	source = Source(('(3, 5)',
					 '(9,)',
					 'swap_top2'))

	code = f.compile(source)

	assert isinstance(code, Code)
	assert len(code.data) == len(source.data)

	ret = f.run(code)

	assert isinstance(ret, Block)
	assert ret.type == Block.type_int_pair
	assert isinstance(ret.data, tuple)
	assert len(ret.data) == 2

	# tuple (good)
	source = Source(('(3, 5, 4, 2)',
					 '(9,)',
					 'swap_top2'))

	code = f.compile(source)

	assert isinstance(code, Code)
	assert len(code.data) == len(source.data)

	ret = f.run(code)

	assert isinstance(ret, Block)
	assert ret.type == Block.type_NESW
	assert isinstance(ret.data, tuple)
	assert len(ret.data) == 4

	# tuple (bad)
	source = Source(('(2, ")',
					 'sto_a'))

	code = f.compile(source)

	assert isinstance(code, Block)
	assert code.type == Block.type_error
	assert code.data.startswith('Malformed tuple')

	# tuple (bad)
	source = Source(('(2, 8, 6)',
					 'sto_a'))

	code = f.compile(source)

	assert isinstance(code, Block)
	assert code.type == Block.type_error
	assert code.data.startswith('Tuple must be')

	# vector (good)
	source = Source(('[3, 5, 4]',
					 '(9,)',
					 'swap_top2'))

	code = f.compile(source)

	assert isinstance(code, Code)
	assert len(code.data) == len(source.data)

	ret = f.run(code)

	assert isinstance(ret, Block)
	assert ret.type == Block.type_vector
	assert isinstance(ret.data, list)
	assert len(ret.data) == 3

	# vector (bad)
	source = Source(('[2, "]',
					 'sto_a'))

	code = f.compile(source)

	assert isinstance(code, Block)
	assert code.type == Block.type_error
	assert code.data.startswith('Malformed vector')

	# Unknown opcode
	source = Source(('sto_a',
					 'make_it_happen'))

	code = f.compile(source)

	assert isinstance(code, Block)
	assert code.type == Block.type_error
	assert code.data.startswith('Unknown opcode')

	# Empty source
	source = Source(tuple())

	code = f.compile(source)

	assert isinstance(code, Block)
	assert code.type == Block.type_error
	assert code.data == 'Empty source'
