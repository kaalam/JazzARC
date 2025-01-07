# Jazz (c) 2018-2025 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses

import json
import random
import shutil

import numpy as np

from Search		import Search
from SearchConf import EVAL_FULL_MATCH
from SearchConf import LENGTH_CODE_EVAL


class MockMCTS:

	def run_search(self, problem, stop_rlz):
		"""
		This mocks the run_search() method of the class MCTS for testing purposes.

		Returns a dictionary:

			'source'	  : A list of (maximum) NUM_TOP_SOLUTIONS as source code to help serialization
 			'evaluation'  : A list of (maximum) NUM_TOP_SOLUTIONS CodeEval np.array.tolist() evaluations for each snippet
			'elapsed'	  : A list of (maximum) NUM_TOP_SOLUTIONS the second each solution was found
			'num_walks'	  : A list of (maximum) NUM_TOP_SOLUTIONS the walk number at which the solution was found
 			'prediction'  : A list of (maximum) NUM_TOP_SOLUTIONS for each code: a list of answers to each question as an np.array.tolist()
			'tot_elapsed' : The total number of seconds the search used
			'tot_walks'	  : The total number of walks the search did
			'stopped_on'  : Either 'time', 'lost' or 'found' the condition that stopped the search
		"""

		answers = []
		for example in problem:
			_, answer, is_test = example.data
			if is_test:
				answers.append(answer)

		source	  	= []
		evaluation	= []
		elapsed		= []
		num_walks	= []
		prediction	= []
		tot_elapsed = 999*random.random()
		tot_walks	= int(55*tot_elapsed)
		stopped_on	= 'time'

		num_sol	= min(int(random.random()*15), 3)
		if num_sol == 0:
			stopped_on	= 'lost'

		for _ in range(num_sol):
			elapsed.append(tot_elapsed)
			num_walks.append(tot_walks)
			tt = 999*random.random()
			tot_elapsed += tt
			tot_walks	+= int(55*tt)

			if random.random() < 0.4:
				source_item = ('cheating',)
				source.append(source_item)
				evaluation.append(np.zeros(LENGTH_CODE_EVAL).tolist())
			else:
				source_item = ('silly_guess',)
				source.append(source_item)
				if random.random() < 0.2:
					evaluation.append(np.full(LENGTH_CODE_EVAL, EVAL_FULL_MATCH).tolist())
					stopped_on = 'found'
				else:
					evaluation.append(np.zeros(LENGTH_CODE_EVAL).tolist())

			for answer in answers:
				pred = []
				if source_item == ('cheating',):
					pred.append(answer.data.tolist())
				else:
					pred.append([[1, 2], [3, 4]])

			prediction.append(pred)

		return {'source' : source, 'evaluation' : evaluation, 'elapsed' : elapsed, 'num_walks' : num_walks,
				'prediction' : prediction, 'tot_elapsed' : tot_elapsed, 'tot_walks' : tot_walks, 'stopped_on' : stopped_on}


def test_experiments():
	random.seed(2001)

	stop_rlz = {
		'broken_threshold'		: 0.1,
		'max_broken_walks'		: 50,
		'max_elapsed_sec'		: 2,
		'min_num_walks'			: 30,
		'stop_num_full_matches' : 3
	}

	x_path = Search.build_experiment(15, 30, stop_rlz)

	with open(x_path + '/config.json', 'r') as f:
		conf = json.load(f)

	assert(len(conf['solved']) == 15)
	assert(len(conf['not_solved']) == 15)
	assert(conf['stop_rlz']['max_elapsed_sec'] == 2)
	assert(conf['stop_rlz']['stop_num_full_matches'] == 3)

	random.seed(2001)
	Search.run_experiment(x_path, mcts=MockMCTS())

	with open(x_path + '/result_details.json', 'r') as f:
		results = json.load(f)

	name = conf['solved'][0]
	assert isinstance(results[name]['correct'][0], int)

	with open(x_path + '/result_summary.txt', 'r') as f:
		summary = f.read().splitlines()

	assert summary[0].startswith('Total number of problems')
	assert summary[1].startswith('Total number of questions')
	assert summary[2].startswith('Total number of correct solutions')
	assert summary[3].startswith('Total running time')

	assert summary[5].startswith('Correct solutions in code base')
	assert summary[6].startswith('New correct solutions')
	assert summary[7].startswith('Failed in code base')
	assert summary[8].startswith('Found and wrong')
	assert summary[9].startswith('New correct found')

	shutil.rmtree(x_path)


def test_show_found_code():
	ret = Search.show_found_code()

	assert isinstance(ret, list)

	# print('\n'.join(ret))
	# print('Done.')
