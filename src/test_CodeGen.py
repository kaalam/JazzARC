# Jazz (c) 2018-2025 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses

import numpy as np

import CodeGen

from Bebop		import Bebop
from Block		import Block
from Context	import Context
from Example	import Example
from MctsNode	import MctsNode
from SearchConf	import REWARD_DISCOUNT
from SearchConf	import LENGTH_CODE_EVAL
from SearchConf	import IDX_PIC_BETTER_MEAN
from SearchConf	import IDX_PIC_REACH_MEAN
from SearchConf	import IDX_PIC_REACH_MIN
from SearchConf	import DUMMY_REWARD_INSTEAD


def test_CodeGen():
	context	= Context('data')
	codegen	= CodeGen.CodeGen('./code_base/basis_code.jcb', './code_base/basis_code.evd', Example)

	problem = context.get_problem_startswith('917bccba.json')		# Hard enough to be unsolved

	codegen.demo_q	 = []
	codegen.demo_a	 = []
	codegen.question = []

	for example in problem:
		q, a, is_test = example.data
		if is_test:
			codegen.question.append(q)
		else:
			codegen.demo_q.append(q)
			codegen.demo_a.append(a)

	root = MctsNode()

	new_moves = codegen.new_moves(root)

	visits	= 0
	rewards	= 0
	for code_item, prior, reward, eval in new_moves:
		assert isinstance(code_item, tuple)

		assert prior  > 0 and prior  <= 2
		assert reward > 0 and reward <= 1

		pr = codegen.predict_rewards(np.stack([eval], axis=0))

		assert pr[0] == reward

		child = MctsNode(code_item, prior, reward, root)

		assert isinstance(child, MctsNode)

		visits	+= 1
		rewards	+= reward

	root.visits += visits
	root.reward += REWARD_DISCOUNT*rewards

	node = root

	while not node.is_leaf():
		node = node.select_child()

	new_moves = codegen.new_moves(node)

	visits	= 0
	rewards	= 0
	for code_item, prior, reward, eval in new_moves:
		assert isinstance(code_item, tuple)

		assert prior  > 0 and prior  <= 2
		assert reward > 0 and reward <= 1

		pr = codegen.predict_rewards(np.stack([eval], axis=0))

		assert pr[0] == reward

		child = MctsNode(code_item, prior, reward, root)

		assert isinstance(child, MctsNode)

		visits	+= 1
		rewards	+= reward


def test_CornerCases():
	codegen	= CodeGen.CodeGen('./code_base/basis_code.jcb', './code_base/basis_code.evd', Example)

	codegen.demo_q	 = [Block.new_picture(list_of_list=[[4, 0, 4]])]
	codegen.demo_a	 = [Block.new_picture(list_of_list=[[5, 0, 5]])]
	codegen.question = []

	code = (Bebop.bebop_get_question,)

	root = MctsNode()
	node = root

	for _ in range(25):
		node = MctsNode(code, 0.9, 0.88, node)

	assert len(codegen.new_moves(node)) == 0

	root = MctsNode()
	node = MctsNode((Block.new_int_pair((2, 4)), Bebop.bebop_get_question), 0.9, 0.88, root)

	assert len(codegen.new_moves(node)) > 0

	codegen.demo_q	 = []
	codegen.demo_a	 = []
	codegen.question = [Block.new_picture(list_of_list=[[5, 0, 5]])]		# No answer -> CodeEval LookupError -> CodeGen returns []

	assert codegen.new_moves(node) == []


def test_AlternativeRewardFunction():
	old_dummy = DUMMY_REWARD_INSTEAD

	CodeGen.DUMMY_REWARD_INSTEAD = 1

	codegen	= CodeGen.CodeGen(None, None, Example)

	codegen.train_reward_function(None)

	eval = np.zeros(LENGTH_CODE_EVAL)

	assert codegen.predict_rewards([eval])[0] == 0

	eval[IDX_PIC_BETTER_MEAN] = 0.5

	assert codegen.predict_rewards([eval])[0] == 0

	eval[IDX_PIC_REACH_MEAN] = 0.6
	eval[IDX_PIC_REACH_MIN]  = 0.6

	assert codegen.predict_rewards([eval])[0] == 0.6

	CodeGen.DUMMY_REWARD_INSTEAD = old_dummy
