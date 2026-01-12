# Jazz (c) 2018-2026 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses

import pytest

from Answer		import Answer
from Bebop		import Bebop
from Block		import Block
from Bond		import Bond
from BopBack	import BopBack
from BopForward	import BopForward
from Code		import Code
from CodeBase	import CodeBase
from CodeGen	import CodeGen
from CodeTree	import CodeTree
from CodeEval	import CodeEval
from Core		import Core
from Container	import Container
from Context	import Context
from Example	import Example
from Field		import Field
from Function	import Function
from MCTS		import MCTS
from MctsNode	import MctsNode
from Problem	import Problem
from Question	import Question
from Search		import Search
from Source		import Source


def test_instance_everything():
	a = Answer([[1, 0], [0, 1]])
	assert isinstance(a, Answer)
	assert Answer.__base__ == Block
	with pytest.raises(AttributeError):
		_ = a.__base__

	a = Bebop()
	assert isinstance(a, Bebop)
	assert Bebop.__base__ == object
	with pytest.raises(AttributeError):
		_ = a.__base__

	a = Block('hello', Block.type_error)
	assert isinstance(a, Block)
	assert Block.__base__ == object
	with pytest.raises(AttributeError):
		_ = a.__base__

	a = Bond(None, None)
	assert isinstance(a, Bond)
	assert Bond.__base__ == Block
	with pytest.raises(AttributeError):
		_ = a.__base__

	a = BopBack()
	assert isinstance(a, BopBack)
	assert BopBack.__base__ == object
	with pytest.raises(AttributeError):
		_ = a.__base__

	a = BopForward()
	assert isinstance(a, BopForward)
	assert BopForward.__base__ == object
	with pytest.raises(AttributeError):
		_ = a.__base__

	a = Code((Function(len, [Block.type_picture], Block.type_picture), Function(len, [Block.type_picture], Block.type_picture)))
	assert isinstance(a, Code)
	assert Code.__base__ == Block
	with pytest.raises(AttributeError):
		_ = a.__base__

	code = a

	a = CodeBase()
	assert isinstance(a, CodeBase)
	assert CodeBase.__base__ == Container
	with pytest.raises(AttributeError):
		_ = a.__base__

	a = CodeEval(Example)
	assert isinstance(a, CodeEval)
	assert CodeEval.__base__ == Field
	with pytest.raises(AttributeError):
		_ = a.__base__

	a = CodeGen(None, None, Example)
	assert isinstance(a, CodeGen)
	assert CodeGen.__base__ == CodeTree
	with pytest.raises(AttributeError):
		_ = a.__base__

	a = CodeTree(None, Example)
	assert isinstance(a, CodeTree)
	assert CodeTree.__base__ == CodeEval
	with pytest.raises(AttributeError):
		_ = a.__base__

	a = Core(code, [Block.new_picture(list_of_list=[[3, 2], [2, 3]])])
	assert isinstance(a, Core)
	assert Core.__base__ == Block
	with pytest.raises(AttributeError):
		_ = a.__base__

	a = Container()
	assert isinstance(a, Container)
	assert Container.__base__ == object
	with pytest.raises(AttributeError):
		_ = a.__base__

	a = Context()
	assert isinstance(a, Context)
	assert Context.__base__ == Container
	with pytest.raises(AttributeError):
		_ = a.__base__

	a = Example([[3, 2], [2, 3]], [[1, 0], [0, 1]], False)
	assert isinstance(a, Example)
	assert Example.__base__ == Bond
	with pytest.raises(AttributeError):
		_ = a.__base__

	a = Field(Example)
	assert isinstance(a, Field)
	assert Field.__base__ == Bebop
	with pytest.raises(AttributeError):
		_ = a.__base__

	a = Function(len, Block.type_nothing, Block.type_nothing)
	assert isinstance(a, Function)
	assert Function.__base__ == Block
	with pytest.raises(AttributeError):
		_ = a.__base__

	a = MCTS(None, None, Example)
	assert isinstance(a, MCTS)
	assert MCTS.__base__ == CodeGen
	with pytest.raises(AttributeError):
		_ = a.__base__

	a = MctsNode()
	assert isinstance(a, MctsNode)
	assert MctsNode.__base__ == object
	with pytest.raises(AttributeError):
		_ = a.__base__

	a = Problem()
	assert isinstance(a, Problem)
	assert Problem.__base__ == Container
	with pytest.raises(AttributeError):
		_ = a.__base__

	a = Question([[3, 2], [2, 3]])
	assert isinstance(a, Question)
	assert Question.__base__ == Block
	with pytest.raises(AttributeError):
		_ = a.__base__

	a = Search()
	assert isinstance(a, Search)
	assert Search.__base__ == object
	with pytest.raises(AttributeError):
		_ = a.__base__

	a = Source(('nop'))
	assert isinstance(a, Source)
	assert Source.__base__ == Block
	with pytest.raises(AttributeError):
		_ = a.__base__
