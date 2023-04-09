# Jazz (c) 2018-2023 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses

import json
import os
import random
import subprocess

import numpy as np

from time import gmtime
from time import strftime

from Block		import Block
from BopBack	import BopBack
from Code		import Code
from CodeBase	import CodeBase
from Context	import Context
from Core		import Core
from Example	import Example
from Field		import Field
from MCTS		import MCTS
from SearchConf	import EVAL_FULL_MATCH
from SearchConf	import IDX_PIC_REACH_MIN
from SearchConf	import IDX_PIC_REACH_MEAN
from Source		import Source


class Search():
	"""
	This is a main (abstract) class for utility purposes. It is NOT part of the submission.

	It is the only one at the top, (CodeTree, TreeNode, CodeGen, CodeEval, MCTS), owning a Context. All
	other top classes, just work on a given problem and are part of the submission. In Kagglespace
	there is no need for a Context.

	This class is the workhorse for testing and developing the other classes over all known problems.
	"""

	@staticmethod
	def build_experiment(num_in_codebase, total_num_problems, stop_rlz):
		"""
		Creates a new folder with problems (num_in_codebase with known solutions and up to total_num_problems) and a
		configuration json with the metaparameters (arguments + content of SearchConf).

		Returns a path that can be used as an argument for run_experiment()
		"""
		assert os.getcwd().endswith('ARC')

		exp_path = './experiments/' + strftime("%Y-%m-%d_%H%M%S", gmtime())

		os.mkdir(exp_path)

		sp = Search.solved_problems()

		solved	   = [name for name in sp.keys()]
		not_solved = [name for name in os.listdir('./data') if name not in solved]

		if len(solved) > num_in_codebase:
			solved = random.sample(solved, num_in_codebase)

		num_not_solved = total_num_problems - len(solved)

		if len(not_solved) > num_not_solved:
			not_solved = random.sample(not_solved, num_not_solved)

		with open('./src/SearchConf.py', 'r') as f:
			search_conf = f.read().splitlines()

		conf = {}

		conf['experiment_path']		 = exp_path
		conf['solved']				 = solved
		conf['not_solved']			 = not_solved
		conf['stop_rlz']			 = stop_rlz
		conf['search_conf']			 = search_conf
		conf['git_commit_long_hash'] = subprocess.check_output('git log --pretty=format:"%H" -n 1', shell=True).decode("utf-8")

		with open(exp_path + '/config.json', 'w') as f:
			json.dump(conf, f)

		return exp_path


	@staticmethod
	def run_experiment(experiment_path, mcts=None):
		"""
		Runs the MCTS.run_search() over all the problems in the experiment:
		  - saves the best codes
		  - verifies the solution and saves the result

		Both result files are updated at each search iteration. You can tail -f the summary to see the progress

		Full details are stored in experiment_path + '/result_details.json' as a dictionary where the key is the problem name and the
		value is another dictionary with:

			(Computed by MCTS)
			'source'	  : A list of (maximum) NUM_TOP_SOLUTIONS as source code to help serialization
 			'evaluation'  : A list of (maximum) NUM_TOP_SOLUTIONS CodeEval np.array.tolist() evaluations for each snippet
			'elapsed'	  : A list of (maximum) NUM_TOP_SOLUTIONS the second each solution was found
			'num_walks'	  : A list of (maximum) NUM_TOP_SOLUTIONS the walk number at which the solution was found
 			'prediction'  : A LoL for each question in the problem and (maximum) NUM_TOP_SOLUTIONS the guess as an np.array.tolist()
			'tot_elapsed' : The total number of seconds the search used
			'tot_walks'	  : The total number of walks the search did
			'stopped_on'  : Either 'time' or 'found' the condition that stopped the search

			(Computed by run_experiment)
			'correct'	  : A list for each question of (int) number of correct guesses for the question

		Summary details are stored in experiment_path + '/result_summary.txt' as a text file (and printed to stdout) as:

			'Total number of problems          : %4i of %4i'
			'Total number of questions         : %4i'
			'Total number of correct solutions : %4i (%3.1f%%)'
			'Total running time                : %4i minutes'
			''
			'Correct solutions in code base    : %4i (%3.1f%%)'
			'New correct solutions             : %4i (%3.1f%%)'
			'Failed in code base               : %s'
			'Found and wrong                   : %s'
			'New correct found                 : %s'
		"""
		mcts = mcts if mcts is not None else MCTS('./code_base/basis_code.jcb', './code_base/basis_code.evd', Example)

		context = Context('data')

		with open(experiment_path + '/config.json', 'r') as f:
			conf = json.load(f)

		filenames = conf['solved'] + conf['not_solved']

		num_prob_todo	   = len(filenames)
		num_prob_done	   = 0
		tot_num_questions  = 0
		tot_correct_sol	   = 0
		tot_running_sec	   = 0
		correct_in_cdb	   = 0
		correct_new		   = 0
		failed_in_codebase = []
		found_and_wrong	   = []
		found_new		   = []

		details = {}

		for filename in filenames:
			problem = context.get_problem_startswith(filename)

			answers = []
			for example in problem:
				_, answer, is_test = example.data
				if is_test:
					answers.append(answer)

			results = mcts.run_search(problem, conf['stop_rlz'])

			results['correct'] = []

			correct	= 0
			found	= False
			for eval, preds in zip(results['evaluation'], results['prediction']):
				if eval[IDX_PIC_REACH_MIN] == EVAL_FULL_MATCH:
					found = True

				for pred, answer in zip(preds, answers):
					if np.array_equal(answer.data, np.array(pred, dtype=np.int32)):
						correct += 1

				results['correct'].append(correct)

			if correct > 0:
				tot_correct_sol += 1
				if filename in conf['solved']:
					correct_in_cdb += 1
				else:
					correct_new += 1
					found_new.append(filename)
			else:
				if filename in conf['solved']:
					failed_in_codebase.append(filename)

				if found:
					found_and_wrong.append(filename)

			tot_num_questions += len(answers)

			details[filename] = results

			num_prob_done	+= 1
			tot_running_sec	+= results['tot_elapsed']

			summary = []

			summary.append('Total number of problems          : %4i (%3.1f%%)' % (num_prob_done, 100*num_prob_done/num_prob_todo))
			summary.append('Total number of questions         : %4i' % tot_num_questions)
			summary.append('Total number of correct solutions : %4i (%3.1f%%)' % (tot_correct_sol, 100*tot_correct_sol/tot_num_questions))
			summary.append('Total running time                : %4i minutes' % int(tot_running_sec/60))
			summary.append('')
			summary.append('Correct solutions in code base    : %4i (%3.1f%%)' % (correct_in_cdb, 100*correct_in_cdb/tot_num_questions))
			summary.append('New correct solutions             : %4i (%3.1f%%)' % (correct_new, 100*correct_new/tot_num_questions))
			summary.append('Failed in code base               : %s' % ', '.join(failed_in_codebase))
			summary.append('Found and wrong                   : %s' % ', '.join(found_and_wrong))
			summary.append('New correct found                 : %s' % ', '.join(found_new))

			with open(experiment_path + '/result_details.json', 'w') as f:
				json.dump(details, f)

			with open(experiment_path + '/result_summary.txt', 'w') as f:
				f.writelines(["%s\n" % line for line in summary])

		return summary


	@staticmethod
	def build_code_base(output_fn, context):
		"""
		Outermost method building everything from scratch.
		"""

		cd_base	= CodeBase()
		field	= Field(Example)
		solved	= Search.solved_problems()

		for name in solved.keys():
			source = solved[name]
			code   = field.compile(source)
			x_code = Code(code.data + (BopBack.bopback_tests_verify_answer,))

			problem = context.get_problem_startswith(name)

			for example in problem:
				core = Core(x_code)

				question, answer, _ = example.data

				q_copy = question.data.copy()
				a_copy = answer.data.copy()

				core.register['question'] = question
				core.register['answer']	  = answer

				*_, ret = core

				assert isinstance(ret, Block)
				assert ret.type == Block.type_integer
				assert isinstance(ret.data, int)

				assert ret.data == 1

				assert np.array_equal(q_copy, core.register['question'].data)
				assert np.array_equal(a_copy, core.register['answer'].data)

			cd_base.add(code, source, name, question)

		cd_base.save(output_fn)


	@staticmethod
	def code_in_experiment(exp_path):
		"""
		Shows all codes in "found and wrong" and "new correct found" for one experiment.
		"""
		with open(exp_path + '/result_details.json', 'r') as f:
			results = json.load(f)

		with open(exp_path + '/result_summary.txt', 'r') as f:
			summary = f.read().splitlines()

		ret = []

		for hea in['Found and wrong                   : ', 'New correct found                 : ']:
			lin = [s for s in summary if s.startswith(hea)][0][36:]

			if len(lin) > 1:
				ret.append('  ' + hea[:17])
				ret.append('')

				for nam in lin.split(', '):
					ret.append('    ' + nam)
					ret.append('')

					for eval, source in zip(results[nam]['evaluation'], results[nam]['source']):
						ret.append('      %5.2f %s' % (eval[IDX_PIC_REACH_MEAN], str(source)))

					ret.append('')

		return ret


	@staticmethod
	def show_found_code(path='./experiments'):
		"""
		Shows all codes in "found and wrong" and "new correct found" for all experiments.
		"""
		experiments = [x[0] for x in os.walk(path)][1:]
		result		= []

		for exp in experiments:
			result.append(exp)
			result.append('')

			result.extend(Search.code_in_experiment(exp))

		return result


	@staticmethod
	def solved_problems():
		"""
		This is where the collection of solved problems is stored as source.
		"""

# NOT IMPLEMENTED:
# ================

# Low hanging
# -----------
# 12997ef3.json
# 12eac192.json
# 1c0d0a4b.json
# 2013d3e2.json
# 21f83797.json
# 2697da3f.json
# 4093f84a.json
# 4852f2fa.json
# 55059096.json
# 58743b76.json 	# Define a new image decomposition, say rulers (where two opposing quadrants are of just one color, rest is trivial)
# 60a26a3e.json
# 62b74c02.json
# 6d0160f0.json
# 6f473927.json
# 90347967.json
# ae58858e.json
# af24b4cc.json
# b94a9452.json
# cd3c21df.json
# d47aa2ff.json
# e7dd8335.json
# ef135b50.json
# ef26cbf6.json
# f0df5ff0.json

# Moving conditions (inhibitory behavior)
# ---------------------------------------
# 6ad5bdfd.json
# 9b4c17c4.json
# cf133acc.json		# With mandatory state (the color being transmitted)
# f3e62deb.json

# Patterns in place
# -----------------
# 0ca9ddb6.json
# 11852cab.json
# 321b1fc6.json
# 4258a5f9.json
# 42918530.json
# 88a10436.json
# 95990924.json
# dc1df850.json

# Pattern matching
# ----------------
# 50f325b5.json
# 7e02026e.json

# Properties of pictures as vectors or pics
# -----------------------------------------
# 017c7c7b.json
# 19bb5feb.json
# 1fad071e.json
# 27a28665.json
# 2a5f8217.json
# 3194b014.json
# 39a8645d.json
# 42a15761.json
# 445eab21.json
# 44f52bb0.json
# 5117e062.json	(also 1x1)
# 54d82841.json
# 5521c0d9.json
# 6c434453.json
# 6e19193c.json
# 6ecd11f4.json
# 77fdfe62.json
# 794b24be.json
# 7c008303.json
# 855e0971.json
# 963e52fc.json
# a61ba2ce.json
# ac2e8ecf.json
# ae4f1146.json
# b0f4d537.json
# cbded52d.json
# cdecee7f.json
# d2abd087.json
# d631b094.json
# e509e548.json
# f8b3ba0a.json

# Relations defined by 1x1 hints
# ------------------------------
# 070dd51e.json
# 1a07d186.json
# 22eb0ac0.json
# 48d8fb45.json
# 56ff96f3.json
# 7d18a6fb.json
# 992798f6.json
# a1570a43.json
# aabf363d.json	(with despeckle)
# af902bf9.json
# b7249182.json
# dc433765.json
# ded97339.json
# e9614598.json
# f5b8619d.json

# Rectangle logic
# ---------------
# 0a1d4ef5.json	(with properties of pictures)
# 23b5c85d.json
# 25094a63.json
# 31adaf00.json
# 32597951.json
# 3eda0437.json
# 6cf79266.json
# 8731374e.json
# 91714a58.json
# a8d7556c.json
# e76a88a6.json

# Despeckle
# ---------
# 0607ce86.json
# 42a50994.json

# Floodfill
# ---------
# 00d62c1b.json
# 67b4a34d.json
# 7447852a.json
# 7c8af763.json
# 83302e8f.json
# 84db8fc4.json
# 9edfc990.json
# aa18de87.json
# e0fb7511.json

# Move layers
# -----------
# bbc9ae5d.json
# feca6190.json

# Dictionary output
# -----------------
# 9110e3c5.json

# Tiling
# ------
# 539a4f51.json
# 8719f442.json
# 8b28cd80.json
# cad67732.json
# 91413438.json
# 93b4f4b3.json

# Reconstruction
# --------------
# 0934a4d8.json
# 1b60fb0c.json
# 1e97544e.json
# 4c5c2cf0.json
# 903d1b4a.json
# 929ab4e9.json
# 9ecd008a.json
# c663677b.json
# e95e3d8e.json
# f9012d9b.json
# f9d67f8b.json

		solved = {'00576224.json' : Source(('get_question',
											'pic_all_as_pat',
											'pat_flip_left_right',
											'pat_as_pic',
											'sto_a',
											'get_a',
											'2pic_cbind',
											'get_a',
											'2pic_cbind',
											'get_question',
											'get_question',
											'2pic_cbind',
											'get_question',
											'2pic_cbind',
											'sto_a',
											'2pic_rbind',
											'get_a',
											'swap_top2',
											'2pic_rbind')),
				  '007bbfb7.json' : Source(('get_question',
				  							'get_question',
											'2pic_multiply')),
				  '0520fde7.json' : Source(('get_question',
											'pic_fork_on_v_axis_as_pics',
				  							'pics_as_2pic',
											'2pic_and_masks_to_1',
											'(1, 2)',
											'swap_top2',
											'pic_intp_recolor')),
				  '0692e18c.json' : Source(('get_question',
				  							'get_question',
				  							'pic_two_col_reverse',
											'2pic_multiply')),
				  '0b148d64.json' : Source(('get_question',
				  							'pic_fork_by_color_as_pics',
											'pics_filter_single_color',
											'pics_as_pic')),
				  '0bb8deee.json' : Source(('(0, 1)',
				  							'get_question',
											'pic_shape_on_auto_grid',
											'pic_intp_recolor',
											'get_question',
				  							'pic_fork_on_auto_grid_as_pics',
											'pics_pic_multiply_as_pic')),
				  '0c786b71.json' : Source(('(0, 0, 3, 0)',
											'(0, 4, 0, 0)',
											'get_question',
					  						'pic_all_as_pat',
											'pat_flip_up_down',
											'pat_flip_left_right',
											'pat_as_pic',
				  							'pic_nesw_extend',
					  						'pic_all_as_pat',
											'sto_a',
											'pat_flip_left_right',
											'get_a',
											'2pat_merge_as_pic',
				  							'pic_nesw_extend',
					  						'pic_all_as_pat',
											'sto_a',
											'pat_flip_up_down',
											'get_a',
											'2pat_merge_as_pic')),
				  '0c9aba6e.json' : Source(('get_question',
					  						'pic_fork_on_h_axis_as_pics',
				  							'pics_as_2pic',
											'2pic_maximum',
											'(0, 8)',
											'swap_top2',
											'pic_intp_recolor',
											'(6, 0)',
											'swap_top2',
											'pic_intp_recolor',
											'(2, 0)',
											'swap_top2',
											'pic_intp_recolor')),
				  '0d3d703e.json' : Source(('get_question',
					  						'(1, 5)',
				  							'swap_top2',
											'pic_intp_swap_colors',
											'(2, 6)',
				  							'swap_top2',
											'pic_intp_swap_colors',
											'(3, 4)',
				  							'swap_top2',
											'pic_intp_swap_colors',
											'(8, 9)',
				  							'swap_top2',
											'pic_intp_swap_colors')),
				  '1190e5a7.json' : Source(('get_question',
				  							'pic_fork_on_auto_grid_as_pics',
											'pics_main_color_as_vec',
											'get_question',
											'pic_shape_on_auto_grid',
											'pic_vec_recolor_each')),
				  '15696249.json' : Source(('get_question',
				  							'pic_filter_axes',
											'get_question',
											'2pic_multiply')),
				  '195ba7dc.json' : Source(('get_question',
					  						'pic_fork_on_v_axis_as_pics',
				  							'pics_as_2pic',
											'2pic_maximum',
											'(7, 1)',
											'swap_top2',
											'pic_intp_recolor')),
				  '1b2d62fb.json' : Source(('get_question',
					  						'pic_fork_on_v_axis_as_pics',
				  							'pics_as_2pic',
				  							'2pic_maximum',
											'(0, 8)',
											'swap_top2',
											'pic_intp_swap_colors',
											'(9, 0)',
											'swap_top2',
											'pic_intp_swap_colors')),
				  '1cf80156.json' : Source(('get_question',
				  							'pic_fork_by_color_as_pics',
											'pics_filter_single_color',
											'pics_as_pic')),
				  '1e0a9b12.json' : Source(('(0, 0, 1, 0)',
				  							'get_question',
											'pic_nesw_gravity',
											'(0, 0, 1, 0)',
											'swap_top2',
											'pic_nesw_gravity',
											'(0, 0, 1, 0)',
											'swap_top2',
											'pic_nesw_gravity',
											'(0, 0, 1, 0)',
											'swap_top2',
											'pic_nesw_gravity',
											'(0, 0, 1, 0)',
											'swap_top2',
											'pic_nesw_gravity')),
				  '1f85a75f.json' : Source(('get_question',
				  							'pic_fork_by_color_as_pics',
											'pics_filter_single_color',
											'pics_as_pic')),
				  '2072aba6.json' : Source(('[[1, 2], [2, 1]]',
				  							'(6,)',
				  							'int_black_box_as_pic',
					  						'2pic_tile_all',
											'(2,)',
											'get_question',
											'pic_int_zoom_in',
											'2pic_recolor_any_rtl')),
				  '25d8a9c8.json' : Source(('(5,)',
											'get_question',
					  						'pic_filter_axes',
											'pic_int_recolor_all')),
				  '25ff71a9.json' : Source(('get_question',
					  						'pic_all_as_pat',
											'(0, 0, 1, 0)',
											'swap_top2',
											'pat_nesw_drag_all',
											'pat_as_pic')),
				  '27f8ce4f.json' : Source(('get_question',
				  							'pic_filter_mostfreq_col',
											'get_question',
											'2pic_multiply')),
				  '2dc579da.json' : Source(('get_question',
					  						'pic_fork_on_auto_grid_as_pics',
				  							'pics_filter_unique_picture_as_pic')),
				  '310f3251.json' : Source(('(0, 1)',
											'(3,)',
											'int_black_box_as_pic',
											'pic_intp_recolor',
					  						'(2,)',
					  						'(1, 0, 0, 1)',
					  						'get_question',
				  							'pic_all_as_pat',
											'pat_nesw_drag_all',
											'pat_as_pic',
											'pic_int_recolor_all',
											'pic_all_as_pat',
											'get_question',
											'pic_all_as_pat',
											'2pat_merge_as_pic',
											'2pic_multiply')),
				  '31d5ba1a.json' : Source(('(1, 6)',
					  						'get_question',
				  							'pic_fork_on_h_axis_as_pics',
											'pics_as_2pic',
											'2pic_xor_masks_to_1',
											'pic_intp_recolor')),
				  '332efdb3.json' : Source(('get_question',
					  						'[[1, 1], [1, 0]]',
											'swap_top2',
											'2pic_tile_all')),
				  '3428a4f5.json' : Source(('get_question',
					  						'pic_fork_on_h_axis_as_pics',
				  							'pics_as_2pic',
				  							'2pic_xor_masks_to_1',
											'(1, 3)',
											'swap_top2',
											'pic_intp_recolor')),
				  '34b99a2b.json' : Source(('get_question',
					  						'pic_fork_on_v_axis_as_pics',
				  							'pics_as_2pic',
				  							'2pic_xor_masks_to_1',
											'(1, 2)',
											'swap_top2',
											'pic_intp_recolor')),
				  '3618c87e.json' : Source(('get_question',
					  						'pic_fork_color_rest_black_as_pics',
											'pics_as_2pic',
											'sto_a',
											'swap_top2',
											'(0, 0, 1, 0)',
											'swap_top2',
											'pic_nesw_gravity',
											'(0, 0, 1, 0)',
											'swap_top2',
											'pic_nesw_gravity',
											'(0, 0, 1, 0)',
											'swap_top2',
											'pic_nesw_gravity',
											'(1, 9)',
											'swap_top2',
											'pic_intp_recolor',
											'get_a',
											'2pic_maximum',
											'(9, 1)',
											'swap_top2',
											'pic_intp_recolor')),
				  '3906de3d.json' : Source(('get_question',
				  							'(1, 0, 0, 0)',
											'swap_top2',
											'pic_nesw_gravity',
				  							'(1, 0, 0, 0)',
											'swap_top2',
											'pic_nesw_gravity',
				  							'(1, 0, 0, 0)',
											'swap_top2',
											'pic_nesw_gravity',
				  							'(1, 0, 0, 0)',
											'swap_top2',
											'pic_nesw_gravity',
				  							'(1, 0, 0, 0)',
											'swap_top2',
											'pic_nesw_gravity',
				  							'(1, 0, 0, 0)',
											'swap_top2',
											'pic_nesw_gravity',
				  							'(1, 0, 0, 0)',
											'swap_top2',
											'pic_nesw_gravity',
				  							'(1, 0, 0, 0)',
											'swap_top2',
											'pic_nesw_gravity',
				  							'(1, 0, 0, 0)',
											'swap_top2',
											'pic_nesw_gravity')),
				  '3af2c5a8.json' : Source(('(0, 4, 0, 0)',
											'get_question',
				  							'pic_nesw_extend',
					  						'pic_all_as_pat',
											'sto_a',
											'pat_flip_left_right',
											'get_a',
											'2pat_merge_as_pic',
											'(0, 0, 3, 0)',
											'swap_top2',
				  							'pic_nesw_extend',
					  						'pic_all_as_pat',
											'sto_a',
											'pat_flip_up_down',
											'get_a',
											'2pat_merge_as_pic')),
				  '3c9b0459.json' : Source(('get_question',
					  						'pic_all_as_pat',
											'pat_flip_up_down',
											'pat_flip_left_right',
											'pat_as_pic')),
				  '423a55dc.json' : Source(('get_question',
				  							'pic_base_height_as_int',
											'get_question',
											'pic_int_slide_rows_west')),
  				  '4347f46a.json' : Source(('get_question',
					  						'pic_outline_4n')),
				  '46442a0e.json' : Source(('get_question',
											'pic_rotate_90ccw',
											'sto_a',
											'pic_rotate_90ccw',
											'sto_b',
											'pic_rotate_90ccw',
				  							'get_question',
											'2pic_cbind',
											'get_b',
				  							'get_a',
											'2pic_cbind',
											'swap_top2',
											'2pic_rbind')),
				  '46f33fce.json' : Source(('get_question',
					  						'(2,)',
					  						'swap_top2',
											'pic_int_zoom_out',
											'(4,)',
					  						'swap_top2',
											'pic_int_zoom_in')),
				  '48131b3c.json' : Source(('get_question',
				  							'pic_two_col_reverse',
				  							'sto_a',
											'get_a',
											'2pic_cbind',
											'sto_a',
											'get_a',
											'2pic_rbind')),
				  '48f8583b.json' : Source(('get_question',
				  							'pic_filter_leastfreq_col',
											'get_question',
											'2pic_multiply')),
				  '496994bd.json' : Source(('get_question',
					  						'pic_all_as_pat',
											'pat_flip_up_down',
											'get_question',
					  						'pic_all_as_pat',
											'2pat_merge_as_pic')),
				  '4be741c5.json' : Source(('get_question',
				  							'pic_distinct_border_colors')),
				  '4c4377d9.json' : Source(('(0, 0, 3, 0)',
				  							'get_question',
					  						'pic_all_as_pat',
											'pat_flip_up_down',
											'pat_as_pic',
				  							'pic_nesw_extend',
					  						'pic_all_as_pat',
											'sto_a',
				  							'pat_flip_up_down',
											'get_a',
											'2pat_merge_as_pic')),
				  '506d28a5.json' : Source(('get_question',
					  						'pic_fork_on_h_axis_as_pics',
				  							'pics_as_2pic',
				  							'2pic_maximum',
											'(1, 3)',
											'swap_top2',
											'pic_intp_recolor',
											'(2, 3)',
											'swap_top2',
											'pic_intp_recolor')),
				  '5582e5ca.json' : Source(('get_question',
				  							'pic_fork_on_auto_grid_as_pics',
											'pics_main_color_as_vec',
											'vec_row_as_pic',
											'get_question',
											'2pic_tile_all')),
				  '5614dbcf.json' : Source(('get_question',
				  							'(4, 5)',
											'swap_top2',
											'pic_intp_swap_colors',
											'(3, 3)',
											'swap_top2',
											'pic_intp_zoom_fit',
											'(4, 0)',
											'swap_top2',
											'pic_intp_recolor',
				  							'(4, 5)',
											'swap_top2',
											'pic_intp_swap_colors')),
				  '5783df64.json' : Source(('(3, 3)',
					  						'get_question',
											'pic_intp_zoom_fit')),
				  '59341089.json' : Source(('get_question',
											'get_question',
				  							'pic_all_as_pat',
											'pat_flip_left_right',
											'pat_as_pic',
											'2pic_cbind',
											'sto_a',
											'get_a',
											'2pic_cbind')),
				  '5b6cbef5.json' : Source(('get_question',
				  							'get_question',
											'2pic_multiply')),
				  '5bd6f4ac.json' : Source(('get_question',
					  						'(0, 6)', '(3, 3)',
				  							'swap_top3',
											'pic_2intp_crop')),
				  '5d2a5c43.json' : Source(('get_question',
					  						'pic_fork_on_v_axis_as_pics',
				  							'pics_as_2pic',
				  							'2pic_maximum',
											'(4, 8)',
											'swap_top2',
											'pic_intp_recolor')),
				  '60c09cac.json' : Source(('get_question',
					  						'(2,)',
					  						'swap_top2',
											'pic_int_zoom_in')),
				  '6150a2bd.json' : Source(('get_question',
					  						'pic_all_as_pat',
											'pat_flip_up_down',
											'pat_flip_left_right',
											'pat_as_pic')),
				  '62c24649.json' : Source(('(0, 0, 3, 0)',
					  						'(0, 3, 0, 0)',
											'get_question',
				  							'pic_nesw_extend',
					  						'pic_all_as_pat',
											'sto_a',
											'pat_flip_left_right',
											'get_a',
											'2pat_merge_as_pic',
				  							'pic_nesw_extend',
					  						'pic_all_as_pat',
											'sto_a',
											'pat_flip_up_down',
											'get_a',
											'2pat_merge_as_pic')),
				  '6430c8c4.json' : Source(('get_question',
					  						'pic_fork_on_h_axis_as_pics',
				  							'pics_as_2pic',
											'2pic_maximum',
											'(0, 3)',
											'swap_top2',
											'pic_intp_recolor',
											'(7, 0)',
											'swap_top2',
											'pic_intp_recolor',
											'(2, 0)',
											'swap_top2',
											'pic_intp_recolor')),
				  '66e6c45b.json' : Source(('(2, 2)',
				  							'(1, 1)',
					  						'get_question',
				  							'pic_2intp_crop',
											'(1,)',
											'swap_top2',
											'pic_int_copy_border',
											'sto_a',
											'pic_corners',
											'get_a',
											'swap_top2',
											'2pic_recolor_any_rtl')),
				  '66f2d22f.json' : Source(('get_question',
				  							'pic_fork_on_v_axis_as_pics',
											'pics_as_2pic',
											'2pic_maximum',
											'(3, 5)',
											'swap_top2',
											'pic_intp_recolor',
											'(2, 5)',
											'swap_top2',
											'pic_intp_recolor',
											'pic_two_col_reverse')),
				  '67a3c6ac.json' : Source(('get_question',
					  						'pic_all_as_pat',
											'pat_flip_left_right',
											'pat_as_pic')),
				  '67e8384a.json' : Source(('(0, 0, 3, 0)',
											'(0, 3, 0, 0)',
											'get_question',
				  							'pic_nesw_extend',
					  						'pic_all_as_pat',
											'sto_a',
											'pat_flip_left_right',
											'get_a',
											'2pat_merge_as_pic',
				  							'pic_nesw_extend',
					  						'pic_all_as_pat',
											'sto_a',
											'pat_flip_up_down',
											'get_a',
											'2pat_merge_as_pic')),
				  '68b16354.json' : Source(('get_question',
					  						'pic_all_as_pat',
											'pat_flip_up_down',
											'pat_as_pic')),
				  '68b67ca3.json' : Source(('(2,)',
					  						'get_question',
											'pic_int_zoom_out')),
				  '6d0aefbc.json' : Source(('(0, 3, 0, 0)',
											'get_question',
				  							'pic_nesw_extend',
					  						'pic_all_as_pat',
											'sto_a',
											'pat_flip_left_right',
											'get_a',
											'2pat_merge_as_pic')),
				  '6ea4a07e.json' : Source(('(8, 2)',
											'(1, 3)',
											'(4, 5)',
					  						'get_question',
											'pic_two_col_reverse',
											'pic_intp_swap_colors',
											'pic_intp_swap_colors',
											'pic_intp_swap_colors')),
				  '6f8cd79b.json' : Source(('(8,)',
					  						'get_question',
											'pic_int_empty_border')),
				  '6fa7a44f.json' : Source(('(0, 0, 3, 0)',
											'get_question',
				  							'pic_nesw_extend',
					  						'pic_all_as_pat',
											'sto_a',
											'pat_flip_up_down',
											'get_a',
											'2pat_merge_as_pic')),
				  '7039b2d7.json' : Source(('get_question',
				  							'pic_fork_on_auto_grid_as_pics',
											'pics_main_color_as_vec',
											'get_question',
											'pic_shape_on_auto_grid',
											'pic_vec_recolor_each')),
				  '72ca375d.json' : Source(('get_question',
					  						'pic_fork_by_color_as_pics',
				  							'pics_filter_v_symmetric',
											'pics_as_pic')),
				  '7468f01a.json' : Source(('get_question',
				  							'(8, 5)',
											'swap_top2',
											'pic_intp_swap_colors',
				  							'(4, 9)',
											'swap_top2',
											'pic_intp_swap_colors',
				  							'(1, 7)',
											'swap_top2',
											'pic_intp_swap_colors',
				  							'pic_fork_by_color_as_pics',
											'pics_as_2pic',
											'pic_all_as_pat',
											'pat_flip_left_right',
											'pat_as_pic',
				  							'(8, 5)',
											'swap_top2',
											'pic_intp_swap_colors',
				  							'(4, 9)',
											'swap_top2',
											'pic_intp_swap_colors',
				  							'(1, 7)',
											'swap_top2',
											'pic_intp_swap_colors')),
				  '746b3537.json' : Source(('get_question',
				  							'pic_distinct_border_colors')),
				  '74dd1130.json' : Source(('get_question',
					  						'pic_transpose')),
				  '780d0b14.json' : Source(('get_question',
					  						'pic_fork_on_auto_grid_as_pics',
											'pics_main_color_as_vec',
											'get_question',
											'pic_shape_on_auto_grid',
											'pic_vec_recolor_each')),
				  '7953d61e.json' : Source(('get_question',
											'pic_rotate_90ccw',
											'sto_a',
											'pic_rotate_90ccw',
											'sto_b',
											'pic_rotate_90ccw',
											'get_a',
				  							'get_question',
											'2pic_cbind',
											'swap_top2',
				  							'get_b',
											'2pic_cbind',
											'swap_top2',
											'2pic_rbind')),
				  '7b7f7511.json' : Source(('get_question',
				  							'pic_autohalves_as_pics',
											'pics_as_2pic')),
				  '7fe24cdd.json' : Source(('get_question',
											'pic_rotate_90ccw',
											'sto_a',
											'pic_rotate_90ccw',
											'sto_b',
											'pic_rotate_90ccw',
				  							'get_question',
											'2pic_cbind',
											'get_b',
				  							'get_a',
											'2pic_cbind',
											'swap_top2',
											'2pic_rbind')),
				  '833dafe3.json' : Source(('get_question',
					  						'get_question',
				  							'pic_all_as_pat',
											'pat_flip_left_right',
											'pat_as_pic',
											'2pic_cbind',
											'sto_a',
											'pic_all_as_pat',
											'pat_flip_up_down',
											'pat_as_pic',
											'get_a',
											'swap_top2',
											'2pic_rbind')),
				  '88a62173.json' : Source(('get_question',
				  							'pic_fork_on_auto_grid_as_pics',
											'pics_filter_unique_picture_as_pic')),
				  '8be77c9e.json' : Source(('(0, 0, 3, 0)',
											'get_question',
				  							'pic_nesw_extend',
					  						'pic_all_as_pat',
											'sto_a',
											'pat_flip_up_down',
											'get_a',
											'2pat_merge_as_pic')),
				  '8d5021e8.json' : Source(('get_question',
					  						'get_question',
					  						'pic_all_as_pat',
											'pat_flip_left_right',
											'pat_as_pic',
											'2pic_cbind',
											'sto_a',
					  						'pic_all_as_pat',
											'pat_flip_up_down',
											'pat_as_pic',
											'sto_b',
											'get_a',
											'2pic_rbind',
											'get_b',
											'2pic_rbind')),
				  '8e2edd66.json' : Source(('get_question',
				  							'pic_two_col_reverse',
				  							'sto_a',
											'get_a',
											'2pic_multiply')),
				  '8f2ea7aa.json' : Source(('get_question',
				  							'pic_fork_by_color_as_pics',
											'pics_as_pic',
											'sto_a',
											'get_a',
											'2pic_multiply')),
				  '9172f3a0.json' : Source(('get_question',
					  						'(3,)',
					  						'swap_top2',
											'pic_int_zoom_in')),
				  '94f9d214.json' : Source(('get_question',
					  						'pic_fork_on_h_axis_as_pics',
				  							'pics_as_2pic',
				  							'2pic_maximum',
											'(0, 2)',
											'swap_top2',
											'pic_intp_recolor',
											'(1, 0)',
											'swap_top2',
											'pic_intp_recolor',
											'(3, 0)',
											'swap_top2',
											'pic_intp_recolor')),
				  '9565186b.json' : Source(('get_question',
				  							'pic_filter_mostfreq_col',
											'(0, 5)',
											'swap_top2',
											'pic_intp_recolor')),
				  '99b1bc43.json' : Source(('get_question',
					  						'pic_fork_on_h_axis_as_pics',
				  							'pics_as_2pic',
				  							'2pic_xor_masks_to_1',
											'(1, 3)',
											'swap_top2',
											'pic_intp_recolor')),
				  '9ddd00f0.json' : Source(('get_question',
				  							'pic_all_as_pat',
											'sto_a',
											'pat_flip_up_down',
											'get_a',
											'2pat_merge_as_pic',
											'get_question',
											'pic_rotate_90ccw',
											'sto_a',
											'2pic_maximum',
											'pic_all_as_pat',
											'sto_a',
											'pat_flip_up_down',
											'get_a',
											'2pat_merge_as_pic')),
				  '9dfd6313.json' : Source(('get_question',
					  						'pic_transpose')),
				  '9f236235.json' : Source(('get_question',
					  						'pic_fork_on_auto_grid_as_pics',
											'pics_main_color_as_vec',
											'get_question',
											'pic_shape_on_auto_grid',
											'pic_vec_recolor_each',
											'pic_all_as_pat',
											'pat_flip_left_right',
											'pat_as_pic')),
				  'a416b8f3.json' : Source(('get_question',
				  							'get_question',
											'2pic_cbind')),
				  'a59b95c0.json' : Source(('(0, 1)',
					  						'get_question',
				  							'pic_all_colors_as_vec',
											'vec_length_as_int',
											'int_black_box_as_pic',
											'pic_intp_recolor',
					  						'get_question',
					  						'2pic_multiply')),
				  'a79310a0.json' : Source(('get_question',
				  							'pic_all_as_pat',
											'(0, 0, 1, 0)',
											'swap_top2',
											'pat_nesw_drag_all',
											'pat_as_pic',
											'(8, 2)',
											'swap_top2',
											'pic_intp_recolor')),
				  'a8610ef7.json' : Source(('get_question',
				  							'pic_all_as_pat',
											'pat_flip_up_down',
											'pat_as_pic',
											'get_question',
											'2pic_and_masks_to_1',
											'(1, 2)',
											'swap_top2',
											'pic_intp_recolor',
											'get_question',
											'(8, 1)',
											'swap_top2',
											'pic_intp_recolor',
											'2pic_maximum',
											'(1, 5)',
											'swap_top2',
											'pic_intp_recolor')),
				  'a87f7484.json' : Source(('get_question',
					  						'(3, 3)',
				  							'swap_top2',
				  							'pic_intp_fork_on_shape_as_pics',
											'pics_filter_unique_pattern_as_pic')),
				  'ac0a08a4.json' : Source(('get_question',
				  							'pic_all_colors_as_vec',
											'vec_length_as_int',
											'get_question',
											'pic_int_zoom_in')),
				  'ad7e01d0.json' : Source(('(5,)',
					  						'get_question',
				  							'pic_int_filter_color',
											'get_question',
											'2pic_multiply')),
				  'b1948b0a.json' : Source(('get_question',
					  						'(2, 6)',
				  							'swap_top2',
											'pic_intp_swap_colors')),
				  'b4a43f3b.json' : Source(('get_question',
				  							'pic_fork_on_h_axis_as_pics',
											'pics_as_2pic',
											'swap_top2',
											'(2,)',
											'swap_top2',
											'pic_int_zoom_out',
											'2pic_multiply')),
				  'b91ae062.json' : Source(('get_question',
				  							'pic_all_colors_as_vec',
											'vec_length_as_int',
											'get_question',
											'pic_int_zoom_in')),
				  'bbb1b8b6.json' : Source(('get_question',
				  							'pic_fork_on_v_axis_as_pics',
											'pics_as_2pic',
											'pic_all_as_pat',
											'swap_top2',
											'pic_all_as_pat',
											'2pat_merge_if_disjoint_as_pic')),
				  'bc4146bd.json' : Source(('get_question',
											'get_question',
				  							'pic_all_as_pat',
											'pat_flip_left_right',
											'pat_as_pic',
											'get_question',
											'2pic_cbind',
											'sto_a',
											'get_a',
											'2pic_cbind',
											'2pic_cbind')),
				  'bd4472b8.json' : Source(('get_question',
				  							'pic_fork_on_auto_grid_as_pics',
											'pics_as_2pic',
											'(0, 1)',
											'swap_top2',
											'pic_intp_recolor',
											'swap_top2',
											'pic_transpose',
											'swap_top2',
											'2pic_recolor_any_rtl',
											'(2, 0, 0, 0)',
											'swap_top2',
											'pic_nesw_extend',
											'pic_all_as_pat',
											'get_question',
											'pic_all_as_pat',
											'2pat_merge_as_pic')),
				  'be03b35f.json' : Source(('get_question',
				  							'pic_fork_by_color_as_pics',
											'pics_as_2pic',
											'2pic_tile_all',
											'pic_rotate_90ccw')),
				  'c3202e5a.json' : Source(('get_question',
				  							'pic_fork_on_auto_grid_as_pics',
											'pics_filter_single_color',
											'pics_as_pic')),
				  'c3e719e8.json' : Source(('get_question',
				  							'pic_filter_mostfreq_col',
											'get_question',
											'2pic_multiply')),
				  'c48954c1.json' : Source(('get_question',
				  							'pic_all_as_pat',
											'pat_flip_left_right',
											'pat_as_pic',
											'sto_a',
											'get_question',
											'2pic_cbind',
											'get_a',
											'2pic_cbind',
				  							'sto_a',
				  							'pic_all_as_pat',
											'pat_flip_up_down',
											'pat_as_pic',
											'sto_b',
											'get_a',
											'2pic_rbind',
											'get_b',
											'2pic_rbind')),
				  'c59eb873.json' : Source(('get_question',
					  						'(2,)',
					  						'swap_top2',
											'pic_int_zoom_in')),
				  'c7d4e6ad.json' : Source(('(0, 1)',
				  							'get_question',
											'pic_intp_select_columns',
											'get_question',
											'(1, 5)',
											'swap_top2',
											'pic_intp_swap_colors',
											'2pic_recolor_any_rtl')),
				  'c9e6f938.json' : Source(('(0, 3, 0, 0)',
											'get_question',
				  							'pic_nesw_extend',
					  						'pic_all_as_pat',
											'sto_a',
											'pat_flip_left_right',
											'get_a',
											'2pat_merge_as_pic')),
				  'c9f8e694.json' : Source(('(0, 1)',
				  							'get_question',
											'pic_intp_select_columns',
				  							'get_question',
											'2pic_recolor_any_rtl')),
				  'ccd554ac.json' : Source(('(0, 1)',
				  							'get_question',
											'pic_intp_recolor',
											'get_question',
											'2pic_multiply')),
				  'cce03e0d.json' : Source(('(2,)',
					  						'get_question',
				  							'pic_int_filter_color',
											'get_question',
											'2pic_multiply')),
				  'ce4f8723.json' : Source(('get_question',
					  						'pic_fork_on_h_axis_as_pics',
				  							'pics_as_2pic',
											'2pic_maximum',
											'(1, 3)',
											'swap_top2',
											'pic_intp_recolor',
											'(2, 3)',
											'swap_top2',
											'pic_intp_recolor')),
				  'ce039d91.json' : Source(('get_question',
				  							'pic_all_as_pat',
											'pat_flip_left_right',
											'pat_as_pic',
											'get_question',
											'2pic_and_masks_to_1',
											'(1, 9)',
											'swap_top2',
											'pic_intp_recolor',
											'get_question',
											'2pic_maximum',
											'(9, 1)',
											'swap_top2',
											'pic_intp_recolor')),
				  'd10ecb37.json' : Source(('get_question',
				  							'pic_outline_4n',
											'(2, 2)',
											'(1, 1)',
											'get_question',
											'pic_2intp_crop',
											'2pic_tile_all')),
				  'd13f3404.json' : Source(('(0, 3, 3, 0)',
											'get_question',
				  							'pic_nesw_extend',
											'pic_all_as_pat',
											'sto_a',
											'(0, 1, 1, 0)',
											'get_a',
											'pat_nesw_drag_all',
											'sto_a',
											'(0, 1, 1, 0)',
											'get_a',
											'pat_nesw_drag_all',
											'sto_a',
											'(0, 1, 1, 0)',
											'get_a',
											'pat_nesw_drag_all',
											'sto_a',
											'(0, 1, 1, 0)',
											'get_a',
											'pat_nesw_drag_all',
											'sto_a',
											'(0, 1, 1, 0)',
											'get_a',
											'pat_nesw_drag_all',
											'2pat_merge',
											'2pat_merge',
											'2pat_merge',
											'2pat_merge',
											'2pat_merge_as_pic')),
				  'd19f7514.json' : Source(('get_question',
					  						'pic_fork_on_h_axis_as_pics',
				  							'pics_as_2pic',
											'2pic_maximum',
											'(4, )',
											'swap_top2',
											'pic_int_recolor_all')),
				  'd23f8c26.json' : Source(('(10, 0)',
					  						'get_question',
				  							'(1, 10)',
				  							'get_question',
				  							'pic_v_axis',
											'pic_intp_recolor',
				  							'2pic_recolor_any_rtl',
											'pic_intp_recolor')),
				  'd406998b.json' : Source(('get_question',
					  						'[[0, 3]]',
				  							'swap_top2',
											'2pic_recolor_any_rtl')),
				  'd4b1c2b1.json' : Source(('get_question',
				  							'pic_all_colors_as_vec',
											'vec_length_as_int',
											'get_question',
											'pic_int_zoom_in')),
				  'd511f180.json' : Source(('get_question',
					  						'(5, 8)',
				  							'swap_top2',
											'pic_intp_swap_colors')),
				  'd56f2372.json' : Source(('get_question',
					  						'pic_fork_by_color_as_pics',
				  							'pics_filter_v_symmetric',
											'pics_as_pic')),
				  'dae9d2b5.json' : Source(('get_question',
					  						'pic_fork_on_v_axis_as_pics',
				  							'pics_as_2pic',
											'2pic_maximum',
											'(3, 6)',
											'swap_top2',
											'pic_intp_recolor',
											'(4, 6)',
											'swap_top2',
											'pic_intp_recolor')),
				  'e133d23d.json' : Source(('get_question',
					  						'pic_fork_on_v_axis_as_pics',
				  							'pics_as_2pic',
				  							'2pic_maximum',
											'(6, 2)',
											'swap_top2',
											'pic_intp_recolor',
											'(8, 2)',
											'swap_top2',
											'pic_intp_recolor')),
				  'e345f17b.json' : Source(('get_question',
					  						'pic_fork_on_v_axis_as_pics',
				  							'pics_as_2pic',
											'2pic_maximum',
											'(4, )',
											'swap_top2',
											'pic_int_recolor_all',
											'pic_two_col_reverse')),
				  'e3497940.json' : Source(('get_question',
											'pic_fork_on_v_axis_as_pics',
											'pics_as_2pic',
				  							'pic_all_as_pat',
											'pat_flip_left_right',
											'pat_as_pic',
											'2pic_maximum')),
				  'e633a9e5.json' : Source(('(1,)',
					  						'get_question',
											'pic_int_copy_border')),
				  'e98196ab.json' : Source(('get_question',
				  							'pic_fork_on_h_axis_as_pics',
											'pics_as_2pic',
											'2pic_maximum')),
				  'e99362f0.json' : Source(('get_question',
				  							'(5, 9)',
											'swap_top2',
											'pic_intp_swap_colors',
											'pic_fork_on_auto_grid_as_pics',
											'pics_maximum_as_pic',
				  							'(5, 9)',
											'swap_top2',
											'pic_intp_swap_colors')),
				  'ed36ccf7.json' : Source(('get_question',
					  						'pic_rotate_90ccw')),
				  'ed98d772.json' : Source(('get_question',
											'pic_rotate_90ccw',
											'sto_a',
											'pic_rotate_90ccw',
											'sto_b',
											'pic_rotate_90ccw',
											'get_a',
				  							'get_question',
											'2pic_cbind',
											'swap_top2',
				  							'get_b',
											'2pic_cbind',
											'swap_top2',
											'2pic_rbind')),
				  'f25fbde4.json' : Source(('get_question',
				  							'pic_fork_by_color_as_pics',
											'pics_as_pic',
											'(2,)',
											'swap_top2',
											'pic_int_zoom_in')),
				  'f25ffba3.json' : Source(('get_question',
					  						'pic_all_as_pat',
											'sto_a',
											'pat_flip_up_down',
											'get_a',
											'2pat_merge_as_pic')),
				  'f2829549.json' : Source(('get_question',
					  						'pic_fork_on_v_axis_as_pics',
				  							'pics_as_2pic',
											'2pic_maximum',
											'(0, 3)',
											'swap_top2',
											'pic_intp_swap_colors',
											'(7, 0)',
											'swap_top2',
											'pic_intp_recolor',
											'(5, 0)',
											'swap_top2',
											'pic_intp_recolor')),
				  'f76d97a5.json' : Source(('get_question',
				  							'pic_two_col_reverse',
											'sto_a',
											'(5, 0)',
											'swap_top2',
											'pic_intp_recolor')),
				  'fafffa47.json' : Source(('get_question',
					  						'pic_fork_on_h_axis_as_pics',
				  							'pics_as_2pic',
				  							'2pic_maximum',
											'(0, 2)',
											'swap_top2',
											'pic_intp_swap_colors',
											'(1, 0)',
											'swap_top2',
											'pic_intp_recolor',
											'(9, 0)',
											'swap_top2',
											'pic_intp_recolor')),
				  'fc754716.json' : Source(('get_question',
				  							'pic_all_colors_as_vec',
											'vec_as_int',
											'get_question',
											'pic_int_empty_border'))}
		return solved
