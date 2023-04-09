# Jazz (c) 2018-2023 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses

import pytest

from Answer	  import Answer
from Bebop	  import Bebop
from Block	  import Block
from Bond	  import Bond
from Example  import Example
from Question import Question


def test_Example():
	with pytest.raises(NotImplementedError):
		_ = Bond.function_classes()

	ret = Example.function_classes()

	assert isinstance(ret, set)

	b = Bebop()

	with pytest.raises(KeyError):
		assert b.opcode['pic_all_as_pat'].ret_type == Block.type_pattern

	for abs_class in ret:
		b.use(abs_class)

	assert b.opcode['pic_all_as_pat'].ret_type == Block.type_pattern
	assert len(b.opcode['2pat_merge_as_pic'].arg_types) == 2
	assert b.opcode['pat_flip_up_down'].ret_type == Block.type_pattern
	assert len(b.opcode['tests_verify_answer'].arg_types) == 1

	ret = Example.from_kind()

	assert ret == Question

	ret = Example.to_kind()

	assert ret == Answer
