# Jazz (c) 2018-2026 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses

from Answer		import Answer
from Block		import Block
from BopForward	import BopForward
from Context	import Context
from Example	import Example
from Function	import Function
from Problem 	import Problem
from Question	import Question


def test_calling():
	ctx = Context('data')

	prob = ctx.get_problem_startswith('496994bd')

	assert isinstance(prob, Problem)
	assert isinstance(prob.data[0], Example)

	q, a, is_test = prob.data[0].data

	assert isinstance(q, Question)
	assert isinstance(a, Answer)
	assert isinstance(is_test, bool)

	bop_fun = BopForward.bopforward_pic_all_as_pat

	assert isinstance(bop_fun, Function)
	assert isinstance(bop_fun.arg_types, list)
	assert bop_fun.ret_type == Block.type_pattern

	ff = bop_fun.data

	assert callable(ff)

	ret = ff(q)

	assert isinstance(ret, Block)
	assert ret.type == Block.type_pattern
	assert isinstance(ret.data, tuple)
	assert len(ret.data) == 2
