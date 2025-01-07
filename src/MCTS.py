# Jazz (c) 2018-2025 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses

import time

from operator import itemgetter

from Block		import Block
from Code		import Code
from CodeGen	import CodeGen
from MctsNode	import MctsNode
from SearchConf	import ADD_EXP_NOISE_EACH
from SearchConf	import EVAL_FULL_MATCH
from SearchConf	import IDX_PIC_REACH_MEAN
from SearchConf	import IDX_PIC_REACH_MIN
from SearchConf	import NUM_TOP_SOLUTIONS
from SearchConf	import REWARD_DISCOUNT
from SearchConf	import WEIGHT_MIN_IN_EVAL


class MCTS(CodeGen):
	"""
	This does the MCTS search.

	One instance of <this> is all you need since it inherits <- CodeGen <- CodeTree <- CodeEval <- Field <- Bebop.

	How this works:
	==============

	This owns the tree, by owning "root : MctsNode" and does forward and backprop walks.

	Forward walks go down the tree calling MctsNode.select_child() until a leaf is found.

	When a leaf is found, it is exploded, all the new moves (secondary structure groups) are returned and evaluated by CodeGen.new_moves().
	This gives code items, priors, rewards and value. Tertiary structure (when applicable) is already included by the CodeGen.
	The priors are used bu UCB search (not updated, but possibly with random noise added at root), the rewards give the learning
	part of the search and the values (a weighted average of eval[IDX_PIC_REACH_MIN] and eval[IDX_PIC_REACH_MEAN]) are used for the top N
	list.

	Backprop walks simply update the tree upwards visits and rewards. Rewards are discounted at each step by REWARD_DISCOUNT to
	favor shorter programs over longer ones.

	A top N list of snippets and solutions is maintained.

	Stopping rules include:

	- number of full matches found
	- minimum number of walks
	- maximum elapsed time
	- maximum number of walks below reward threshold
	- minimum reward threshold

	"""

	def __init__(self, code_base_fn, reward_data_fn, abstract_relation):
		CodeGen.__init__(self, code_base_fn, reward_data_fn, abstract_relation)


	def run_search(self, problem, stop_rlz, root=None):
		"""
		Starts a new problem (demo, question) and runs MCTS searches until a stopping condition is matched by either of
		(min_num_walks, stop_num_full_matches, max_broken_walks, broken_threshold, max_elapsed_sec) in that priority.

		1. min_num_walks is done anyway, no stopping before that.
		2. stop_num_full_matches applies only when min_num_walks is complete and breaks without waiting for the timer.
		3. max_broken_walks and broken_threshold: The threshold measures extremely low rewards and lack of progress. if that happens,
		   max_broken_walks can interrupt the search returning 'lost'
		4. max_elapsed_sec will break the search after that many seconds in case not enough solutions are found.

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
		# Init the problem for the whole search
		self.demo_q	  = []
		self.demo_a	  = []
		self.question = []

		for example in problem:
			q, a, is_test = example.data
			if is_test:
				self.question.append(q)
			else:
				self.demo_q.append(q)
				self.demo_a.append(a)

		# Define the Top N list logic
		top_N	   = []
		top_min_ev = 0

		def try_push_to_topN(code_item, eval):
			nonlocal top_N
			nonlocal top_min_ev

			ev = eval[IDX_PIC_REACH_MEAN]*(1 - WEIGHT_MIN_IN_EVAL) + eval[IDX_PIC_REACH_MIN]*WEIGHT_MIN_IN_EVAL
			if ev <= top_min_ev:
				return

			code = Code(self.code_in_path_to_node + code_item)

			top = (ev, code, eval, num_walks, time.time() - start_time)

			if len(top_N) == NUM_TOP_SOLUTIONS:
				top_N.pop()

			top_N.append(top)
			top_N = sorted(top_N, key=itemgetter(0), reverse=True)

			if len(top_N) == NUM_TOP_SOLUTIONS:
				top_min_ev = top_N[-1][0]

		# Define the stopping logic
		ret = {'elapsed' : [], 'evaluation' : [], 'num_walks' : [], 'prediction' : [], 'source' : []}

		def continue_running():
			if num_walks < stop_rlz['min_num_walks']:
				return True

			if num_broken_walks > stop_rlz['max_broken_walks']:
				ret['stopped_on'] = 'lost'
				return False

			if num_solved >= stop_rlz['stop_num_full_matches']:
				ret['stopped_on'] = 'found'
				return False

			if time.time() - start_time > stop_rlz['max_elapsed_sec']:
				ret['stopped_on'] = 'time'
				return False

			return True

		# Init the search/stopping variables
		if root is None:
			root = MctsNode()

		num_solved		 = 0
		num_broken_walks = 0
		num_walks		 = 0
		start_time		 = time.time()

		while continue_running():
			if (num_walks + 1) % ADD_EXP_NOISE_EACH == 0:
				root.add_exploration_noise()

			node = root

			while not node.is_leaf():
				node = node.select_child()

			new_moves = self.new_moves(node)

			if len(new_moves) == 0:
				num_broken_walks += 1
				visits			  = 1
				rewards			  = 0
				solved			  = 0
			else:
				visits	= 0
				rewards	= 0
				solved	= 0
				for code_item, prior, reward, eval in new_moves:
					MctsNode(code_item, prior, reward, node)

					try_push_to_topN(code_item, eval)

					if eval[IDX_PIC_REACH_MIN] == EVAL_FULL_MATCH:
						solved += 1

					visits	+= 1
					rewards	+= reward

				if rewards/visits < stop_rlz['broken_threshold']:
					num_broken_walks += 1
				else:
					num_broken_walks  = 0

			while True:
				node.reward += rewards
				rewards		*= REWARD_DISCOUNT
				node.visits += visits
				if node.is_root():
					break

				node = node.parent

			num_solved += solved
			num_walks  += 1

		ret['tot_elapsed'] = time.time() - start_time
		ret['tot_walks']   = num_walks

		for _, code, eval, num_walks, elapsed in top_N:
			ret['evaluation'].append(eval.tolist())
			ret['elapsed'].append(elapsed)
			ret['num_walks'].append(num_walks)
			ret['source'].append(self.decompile(code, pretty=False))

			self.multicore_clear()

			ret_code = self.multicore_run_all(code)

			assert ret_code.type == Block.type_no_error

			pred = []
			for i, pic in enumerate(self.multicore_state['pic_lists']):
				if i >= len(self.demo_q):
					pred.append(pic[-1].data.tolist())

			ret['prediction'].append(pred)

		return ret
