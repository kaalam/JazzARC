# Jazz (c) 2018-2025 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses

import numpy as np

from Block		import Block
from Context	import Context
from Example	import Example
from MCTS		import MCTS
from MctsNode	import MctsNode
from Problem	import Problem
from SearchConf	import IDX_PIC_REACH_MIN
from SearchConf	import EVAL_FULL_MATCH
from Source		import Source


def test_MctsNode():
	root = MctsNode()

	assert root.prior == 0


def test_MCTS():
	context = Context('data')
	mcts = MCTS('./code_base/basis_code.jcb', './code_base/basis_code.evd', Example)

	stop_rlz = {
		'broken_threshold'		: 0.1,
		'max_broken_walks'		: 50,
		'max_elapsed_sec'		: 2,
		'min_num_walks'			: 5,
		'stop_num_full_matches' : 1
	}

	prob1a = context.get_problem_startswith('ed36ccf7')		# One item
	prob1b = context.get_problem_startswith('007bbfb7')		# One item
	prob2  = context.get_problem_startswith('0692e18c')		# Two items
	probX  = context.get_problem_startswith('c6e1b8da')		# Nightmare task

	answer = None
	for example in prob1a:
		_, a, is_test = example.data
		if is_test:
			answer = a

	ret = mcts.run_search(prob1a, stop_rlz)

	assert ret['stopped_on'] == 'found' and ret['tot_walks'] == 5 and ret['tot_elapsed'] < 10

	N = 0
	for src, evl, ela, n_w, prd in zip(ret['source'], ret['evaluation'], ret['elapsed'], ret['num_walks'], ret['prediction']):
		code = mcts.compile(Source(src))

		assert code.type == Block.type_no_error

		assert evl[IDX_PIC_REACH_MIN] == EVAL_FULL_MATCH

		assert ela < 10

		assert n_w <= 5

		for pic in prd:
			assert np.array_equal(np.array(pic, dtype=np.int32), answer.data)

		N += 1

	assert N == 3

	ret = mcts.run_search(prob1b, stop_rlz)

	assert ret['stopped_on'] == 'found' and ret['tot_walks'] == 5 and ret['tot_elapsed'] < 10

	answer = None
	for example in prob2:
		_, a, is_test = example.data
		if is_test:
			answer = a

	root = MctsNode()

	ret = mcts.run_search(prob2, stop_rlz, root=root)

	assert ret['tot_walks'] < 100 and ret['tot_elapsed'] < 10

	N = 0
	for src, evl, ela, n_w, prd in zip(ret['source'], ret['evaluation'], ret['elapsed'], ret['num_walks'], ret['prediction']):
		code = mcts.compile(Source(src))

		assert code.type == Block.type_no_error
		assert ela < 10
		assert n_w <= 100

		N += 1*(evl[IDX_PIC_REACH_MIN] == EVAL_FULL_MATCH)

		assert ret['stopped_on'] == 'time' or N > 0

	root.print(field=mcts)
	last = root.children[-1]
	last.print()

	ret = mcts.run_search(probX, stop_rlz)

	assert ret['stopped_on'] == 'time' and ret['tot_walks'] < 100 and ret['tot_elapsed'] < 10

	stop_rlz['broken_threshold'] = 1
	stop_rlz['max_broken_walks'] = 4

	ret = mcts.run_search(probX, stop_rlz)

	assert ret['stopped_on'] == 'lost' and ret['tot_walks'] < 100 and ret['tot_elapsed'] < 10


def test_corners():
	mcts = MCTS('./code_base/basis_code.jcb', './code_base/basis_code.evd', Example)

	stop_rlz = {
		'broken_threshold'		: 1,
		'max_broken_walks'		: 3,
		'max_elapsed_sec'		: 2,
		'min_num_walks'			: 1,
		'stop_num_full_matches' : 1
	}

	problem = Problem()

	problem.add(Example([[4, 0, 4]], None, True))

	ret = mcts.run_search(problem, stop_rlz)

	assert ret['stopped_on'] == 'lost' and ret['tot_walks'] <= 10 and ret['tot_elapsed'] < 10
