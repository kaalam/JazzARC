# Jazz (c) 2018-2025 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses

import pytest

from Bebop import Bebop
from Block import Block
from Code  import Code
from Core  import Core


def test_Bebop():
	b = Bebop()

	assert isinstance(b, Bebop)
	assert Bebop.__base__ == object

	with pytest.raises(KeyError):
		b.use(Bebop)

	b = Bebop()

	code = Code((b.opcode['swap_top2'], ))
	core = Core(code, stack=[Block.new_integer(19), Block.new_integer(21)])

	assert core.all_right
	assert core.stack[-1].data == 21

	ret, = core

	assert core.all_right
	assert core.stack[-1].data == 19

	assert isinstance(ret, Block)
	assert ret.type == Block.type_integer
	assert ret.data == 19

	core = Core(code)

	ret, = core

	assert core.all_right is False
	assert core.error_msg == 'swap_top2() with less than two'

	assert isinstance(ret, Block)
	assert ret.type == Block.type_error
	assert ret.data == 'swap_top2() with less than two'

	code = Code((b.opcode['get_answer'], ))
	core = Core(code)

	core.register['answer'] = Block.new_integer(37)

	ret, = core

	assert core.all_right
	assert core.stack[-1].data == 37

	assert isinstance(ret, Block)
	assert ret.type == Block.type_integer
	assert ret.data == 37
