# Jazz (c) 2018-2025 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses

import pytest

from Block	  import Block
from CodeBase import CodeBase
from Example  import Example
from Field	  import Field
from Source	  import Source


def test_successful_end_to_end():
	cb = CodeBase()

	assert cb.num_items() == 0

	f = Field(Example)

	source = Source(('[[1, 0, 0], [0, 2, 2], [0, 0, 0]]',
					 'sto_a',
					 'get_a'))

	code = f.compile(source)

	cb.add(code, source, 'trivial_store_pic', Block.new_picture(list_of_list=[[4, 5], [5, 4]]))

	assert cb.num_items() == 1

	source = Source(('(3, 5)',
					 'sto_a'))

	code = f.compile(source)

	cb.add(code, source, 'trivial_store_tuple', Block.new_picture(list_of_list=[[4, 0, 4]]))

	assert cb.num_items() == 2

	cb.save('test_codebase.jcb')

	base = CodeBase('test_codebase.jcb', Example)

	assert base.num_items() == 2

	N = 0
	for code, source, name, sample in base:
		assert code.type == Block.type_no_error
		assert 'sto_a' in source.data
		assert name.startswith('trivial_')
		assert sample.type == Block.type_picture

		N += 1

	assert N == 2

	code = base.code_by_name('trivial_store_tuple')

	assert code.data[0].type == Block.type_int_pair
	assert code.data[1].type == Block.type_function

	code = base.code_by_name('trivial_store_pic')

	assert len(code.data) == 3

	source = base.source_by_name('trivial_store_tuple')

	assert source.data[0] == '(3, 5)'

	source = base.source_by_name('trivial_store_pic')

	assert len(source.data) == 3

	sample = base.sample_by_name('trivial_store_tuple')

	assert sample.data.shape == (1, 3)

	sample = base.sample_by_name('trivial_store_pic')

	assert sample.data.shape == (2, 2)


def test_all_the_corners():
	f = Field(Example)

	source = Source(('[[1, 0, 0], [0, 2, 2], [0, 0, 0]]',
					 'sto_a',
					 'get_a'))

	code = f.compile(source)

	cb = CodeBase()
	with pytest.raises(NameError):
		cb.add(code, source, 404, Block.new_picture(list_of_list=[[4, 0, 4]]))

	cb = CodeBase()
	with pytest.raises(NameError):
		cb.add(code, source, 'xy', Block.new_picture(list_of_list=[[4, 0, 4]]))

	cb = CodeBase()

	cb.add(code, source, 'nice_name', Block.new_picture(list_of_list=[[4, 0, 4]]))
	assert cb.num_items() == 1

	with pytest.raises(NameError):
		cb.add(code, source, 'nice_name', Block.new_picture(list_of_list=[[4, 0, 4]]))

	with open('test_valid_all.jcb', 'w') as f:
		f.write('.bopDB: xx\n\n\naaa\n---\nget_a\n\n[[4,0,4]]\n\n.eof.')

	with open('test_invalid_head.jcb', 'w') as f:
		f.write('.NodeBase: xx\n\n\naaa\n---\nget_a\n\n[[4,0,4]]\n\n.eof.')

	with open('test_invalid_code.jcb', 'w') as f:
		f.write('.bopDB: xx\n\n\naaa\n---\nmutilate_stack_top\n\n[[4,0,4]]\n\n.eof.')

	with open('test_invalid_name.jcb', 'w') as f:
		f.write('.bopDB: xx\n\n\naa\n--\nget_a\n\n[[4,0,4]]\n\n.eof.')

	with open('test_missing_sample.jcb', 'w') as f:
		f.write('.bopDB: xx\n\n\naa\n--\nget_a\n\n.eof.')

	with open('test_invalid_ending.jcb', 'w') as f:
		f.write('.bopDB: xx\n\n\naaa\n---\nget_a\n\n[[4,0,4]]\n\n.fof.')

	cb = CodeBase('test_valid_all.jcb', Example)

	assert cb.num_items() == 1

	with pytest.raises(ValueError):
		CodeBase('test_invalid_head.jcb', Example)

	with pytest.raises(ValueError):
		CodeBase('test_invalid_code.jcb', Example)

	with pytest.raises(NameError):
		CodeBase('test_invalid_name.jcb', Example)

	with pytest.raises(ValueError):
		CodeBase('test_missing_sample.jcb', Example)

	with pytest.raises(ValueError):
		CodeBase('test_invalid_ending.jcb', Example)
