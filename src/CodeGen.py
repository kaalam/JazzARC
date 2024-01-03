# Jazz (c) 2018-2024 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses

import random

from operator import itemgetter

import numpy as np

from xgboost import XGBClassifier

from Bebop		import Bebop
from Block		import Block
from Code		import Code
from CodeTree	import CodeTree
from SearchConf	import DUMMY_REWARD_INSTEAD
from SearchConf	import EVAL_FULL_MATCH
from SearchConf	import IDX_PIC_BETTER_MEAN
from SearchConf	import IDX_PIC_REACH_MAX
from SearchConf	import IDX_PIC_REACH_MEAN
from SearchConf	import IDX_PIC_REACH_MIN
from SearchConf	import IDX_PIC_WORSE_MEAN
from SearchConf	import LENGTH_CODE_EVAL
from SearchConf	import MAX_MOVES_AT_ROOT
from SearchConf	import NUM_MOVES_STP_DISCOUNT
from SearchConf	import PRIOR_BOOST_IN_SNIPPET
from SearchConf	import WEIGHT_MIN_IN_EVAL
from SearchConf	import XGB_ETA
from SearchConf	import XGB_GAMMA
from SearchConf	import XGB_MAX_DELTA_STEP
from SearchConf	import XGB_MAX_DEPTH
from SearchConf	import XGB_MIN_CHILD_WEIGHT
from SearchConf	import XGB_N_ESTIMATORS
from SearchConf	import XGB_REG_ALPHA
from SearchConf	import XGB_REG_LAMBDA
from SearchConf	import XGB_SUBSAMPLE
from SearchConf	import XGB_VERBOSE


class CodeGen(CodeTree):
	"""
	This class generates legal (== running on all questions and demos) secondary structure code items for nodes.

	The only method used is `(moves, priors) = new_moves(node)`, anything else is internal. Demos and questions are set by MCTS()
	as self.demo_q, self.demo_a, self.question.

	One instance of MCTS is all you need since it inherits <- <this> <- CodeTree <- CodeEval <- Field <- Bebop.

	Random ideas
	============

	Reward function:
	----------------

	The reward function is used to guide search only and avoid the scarcity of positive results by rewarding improvement.

	It is only a function of the evaluation vector (see below) only. It comes from the 1,0 (win/loss), but since it is computed
	by a classifier, it returns the confidence returned by the classifier. It is 1-based in the sense that it will be compared to 1
	for the number of visits.

	Old ideas combine (using variables from SearchConf as weights to simplify experiments):

		- (better > 0 and worse == 0) at both pic and pattern (with weights)
		- a small weight for eval[PIC_REACH_AVERAGE] as VALUE_FULL_MATCH >> 1 the number can get very big with more full solutions
		- negative for wrong size
		- negative when output information is below target information (as in you can recolor 3 cols to 2, but not 2 to 3)

	Now, it is implemented using XGBClassifier learned on a dataset created when building CodeGen of rewards that convert into wins and
	a disjoint set of rewards that do not (maybe they do, but the paths are unknown).

	Code has structure
	------------------

	It is clear that for code generation to be successfull it requires some kind of (protein like):

	primary structure	- Opcode level. This is strongly enforced by typing and could (worst case) need some fast search (e.g., dynamic
						  programming). Code generation must be thought of as in secondary structure.
	secondary structure - Short **sentences** of bytecodes calles code items returning a picture and hence an evaluation and a reward.
	tertiary structure	- Closers (like an automatically identified recoloring sequence of say 5 colors to 1), dictionaries of pattern to
						  output of few categories, zoom_in, zoom_out, macros like "copy and move" possibly adding a "then merge" step to
						  the closer, ... Closers are tricky and will be implemented over time. They leverage the internal representation
						  of the CodeEval (stack, registers, image collections, etc.)

	In other words: primary structure matches types. This could possibly insert auxiliary opcodes, but we are not exploring this yet,
	for the moment we consider using code items and, possibly, modify them with tertiary structure. Secondary structure makes sentences
	that are evaluated and learned by cutting human created programs using a logic explained in CodeTree. Tertiary structure is concerned
	about not-yet-complete snippets end-to-end (more like current position to end which is what is required).

	- The search works at a level of secondary structure.
	- Secondary structure code items have a measureable output, a pic and are taken from a collection built by CodeTree.
	- Tertiary structure looks for clues on possible things to append or change. E.g., the pattern is okay, the pic is not but a simple
	  recolor gets it right, the recolor is added. Or a data constant can be tried with another value. Tertiary structure is applied
	  only once when CodeGen is done (a node is expanded) and can also become an alternative move instering two slightly different items
	  instead of just one.

	Collections:
	------------

	While computing the evaluation, the cores will store to separate push only stacks (picture, pattern and pictures) for
	the last opcode in each snippet, according to output type. Also, the content of non-empty registers {a-e, question}.
	This is also returned so the CodeGen can find (shape adjustments, repetitions, recolor closers, etc.).

	State is NOT (just) a picture (Idea for the future if we us NNs for code generation)
	-----------------------------

	(See appendix E of the MuZero paper). State is a collection of things including:

	- The whole program
	- The evaluation function described in CodeEval

	You have to get shape right first
	---------------------------------

	Whenever possible, the first returned a picture in any program MUST return the correct output shape. Since it is only one sentence,
	it is easy to reverse engineer and suggest starting instructions like: pic_fork_on_v_axis_as_pics or (0, 3, 3, 0), pic_nesw_extend.
	Obviously, once right it cannot be changed. This is favored by the eval function (see CodeEval).

	Update: Now we have empirical data for the rewards based on actual solutions so the importance of getting shape right asap will be
	backed up by data. Maybe it is no so important. E.g., the pics resulting from some transformation + zoom in will no work if you zoom
	in too early.
	"""

	def __init__(self, code_base_fn, reward_data_fn, abstract_relation):
		CodeTree.__init__(self, code_base_fn, abstract_relation)

		if reward_data_fn is not None:
			self.train_reward_function(reward_data_fn)


	def build_reward_training_data(self, save_as_fn, context):
		"""
		Builds a dataset of rewards at the end of each code item for solved and unsolved problems used to train a classifier.

		(Called during testing only. The dataset is persisted, because the dependency on context is not acceptable in kagglespace.)
		"""

		get_or_swap = set([Bebop.bebop_get_question, Bebop.bebop_get_a, Bebop.bebop_get_b, Bebop.bebop_get_c, Bebop.bebop_get_d,
						   Bebop.bebop_get_e, Bebop.bebop_swap_top2, Bebop.bebop_swap_top3])

		sto = set([Bebop.bebop_sto_a, Bebop.bebop_sto_b, Bebop.bebop_sto_c, Bebop.bebop_sto_d, Bebop.bebop_sto_e])

		data_str = []

		for code, _, name, _ in self.cb:
			problem = context.get_problem_startswith(name)

			self.demo_q		 = []
			self.demo_a		 = []
			self.question	 = []

			for example in problem:
				q, a, is_test = example.data
				if is_test:
					self.question.append(q)
				else:
					self.demo_q.append(q)
					self.demo_a.append(a)

			self.multicore_clear()

			stack = self.multicore_state['stacks'][0]

			l_stack_npic = 0
			last_op		 = None

			for opcode in code.data:
				ret = self.multicore_run_all(Code((opcode,)), ignore_ret_type=True)

				assert ret.type == Block.type_no_error

				stack_npic = 0
				for blk in reversed(stack):
					if blk.type != Block.type_picture:
						break
					stack_npic += 1

				if l_stack_npic > 0 and opcode not in sto and last_op not in get_or_swap and last_op.type == Block.type_function:
					data_str.append('1, ' + str(self.eval_code().tolist()).replace('[', '').replace(']', ''))

				l_stack_npic = stack_npic
				last_op		 = opcode

			ev = self.eval_code()
			data_str.append('1, ' + str(ev.tolist()).replace('[', '').replace(']', ''))

			assert ev[IDX_PIC_REACH_MIN] == EVAL_FULL_MATCH

		N = context.num_items()

		random.seed(2001)

		for code, _, name, _ in self.cb:
			problems = []
			for _ in range(2):
				problems.append(context[random.randrange(N)])

			for problem in problems:
				self.demo_q		 = []
				self.demo_a		 = []
				self.question	 = []

				for example in problem:
					q, a, is_test = example.data
					if is_test:
						self.question.append(q)
					else:
						self.demo_q.append(q)
						self.demo_a.append(a)

				self.multicore_clear()

				stack = self.multicore_state['stacks'][0]

				l_stack_npic = 0
				last_op		 = None
				broken		 = False

				for opcode in code.data:
					ret = self.multicore_run_all(Code((opcode,)), ignore_ret_type=True)

					if ret.type != Block.type_no_error:
						broken = True
						break

					stack_npic = 0
					for blk in reversed(stack):
						if blk.type != Block.type_picture:
							break
						stack_npic += 1

					if l_stack_npic > 0 and opcode not in sto and last_op not in get_or_swap and last_op.type == Block.type_function:
						try:
							ev = self.eval_code()
						except LookupError:
							broken = True

						if not broken:
							if ev[IDX_PIC_REACH_MAX] < EVAL_FULL_MATCH:
								data_str.append('0, ' + str(ev.tolist()).replace('[', '').replace(']', ''))

					l_stack_npic = stack_npic
					last_op		 = opcode

				if not broken:
					try:
						ev = self.eval_code()
					except LookupError:
						broken = True

					if not broken:
						if ev[IDX_PIC_REACH_MAX] < EVAL_FULL_MATCH:
							data_str.append('0, ' + str(ev.tolist()).replace('[', '').replace(']', ''))

		with open(save_as_fn, 'w') as f:
			f.write('.priorTRN: ' + save_as_fn + '\n')

			for st in data_str:
				f.write(st + '\n')

			f.write('.eof.')


	def train_reward_function(self, data_fn):
		"""
		Trains an XGBClassifier on the data to learn predicting reward.

		(Called during construction of the object only.)
		"""
		if DUMMY_REWARD_INSTEAD:
			return

		with open(data_fn, 'r') as f:
			txt = f.read().splitlines()

		lol = []

		for s in txt:
			if s[0] != '.':
				lol.append([float(x) for x in s.split(', ')])

		train = np.array(lol)

		self.xgb = XGBClassifier(n_estimators	  = XGB_N_ESTIMATORS,
								 eta			  = XGB_ETA,
								 gamma			  = XGB_GAMMA,
								 max_depth		  = XGB_MAX_DEPTH,
								 min_child_weight = XGB_MIN_CHILD_WEIGHT,
								 max_delta_step	  = XGB_MAX_DELTA_STEP,
								 subsample		  = XGB_SUBSAMPLE,
								 reg_lambda		  = XGB_REG_LAMBDA,
								 reg_alpha		  = XGB_REG_ALPHA)

		self.xgb.fit(train[:, range(1, LENGTH_CODE_EVAL + 1)], train[:, 0], verbose=XGB_VERBOSE)


	def predict_rewards(self, evaluations):
		"""
		Uses the trained XGBClassifier to predict rewards for the evaluation of the new moves created.
		"""
		if DUMMY_REWARD_INSTEAD:
			ret = []
			for eval in evaluations:
				if eval[IDX_PIC_BETTER_MEAN] > eval[IDX_PIC_WORSE_MEAN]:
					ev = eval[IDX_PIC_REACH_MEAN]*(1 - WEIGHT_MIN_IN_EVAL) + eval[IDX_PIC_REACH_MIN]*WEIGHT_MIN_IN_EVAL
				else:
					ev = 0
				ret.append(ev)

			return np.array(ret)

		return self.xgb.predict_proba(evaluations)[:, 1]


	def new_moves(self, node):
		"""
		Returns legal (== running on all the questions) secondary structure items with their prior and reward.

		Returns list of (item, prior, reward, eval) tuples. And sets the property self.code_in_path_to_node with the code run before
		the code in the node.
		"""
		path_to_node = []
		nd_up		 = node
		node_height	 = 0
		max_moves	 = MAX_MOVES_AT_ROOT

		while not nd_up.is_root():
			path_to_node.append(nd_up)

			nd_up		 = nd_up.parent
			node_height += 1
			max_moves	 = int(max_moves*NUM_MOVES_STP_DISCOUNT)

		if max_moves < 1:
			return []

		self.multicore_clear()

		self.code_in_path_to_node = ()
		for nd_down in reversed(path_to_node):
			self.code_in_path_to_node = self.code_in_path_to_node + nd_down.code

		if self.code_in_path_to_node != ():
			# if self.hash_code(node.code) in ['1ed6a8be2d6e44f0f477b312622455cb', 'de0e344a655966aca556682e21b93ca6']:
			# 	self.hashfound = True
			# 	pass

			# if self.hash_code(node.code) == '1ed6a8be2d6e44f0f477b312622455cb' and node.parent.code is not None:
			# 	if self.hash_code(node.parent.code) == 'de0e344a655966aca556682e21b93ca6':
			# 		self.hashfound = True
			# 		pass

			ret = self.multicore_run_all(Code(self.code_in_path_to_node))

			# if ret.type != Block.type_no_error:
			# 	print('node:', hex(id(node)))
			# 	print('node.parent:', hex(id(node.parent)))
			# 	print('node.parent.parent:', hex(id(node.parent.parent)))
			# 	print('-------------------')
			# 	print(ret.data)
			# 	print('-------------------')
			# 	print('node.code hash', self.hash_code(node.code))
			# 	print('node.parent.code hash', self.hash_code(node.parent.code))
			# 	print('-------------------')
			# 	nd_up.print(field=self)

			assert ret.type == Block.type_no_error

		state_to_node = self.multicore_copy_state(self.multicore_state)

		stack = self.multicore_state['stacks'][0]

		stack_height = len(stack)

		stack_npic = 0
		for blk in reversed(stack):
			if blk.type != Block.type_picture:
				break
			stack_npic += 1

		code_priors = []

		uses = self.item_prior_by_stackuse.keys()
		for use in uses:
			npic, depth = self.stack_npic_depth(use)
			if npic <= stack_npic and depth <= stack_height:			# The stack frame is compatible with the use
				code_priors.extend(self.item_prior_by_stackuse[use])

		code_priors = sorted(code_priors, key=itemgetter(1), reverse=True)

		hash_prev_items = self.hash_code(self.code_in_path_to_node)

		items  = []
		priors = []
		evals  = []
		for item, prior in code_priors:
			self.multicore_set_state(state_to_node)

			# if self.hash_code(item) in ['1ed6a8be2d6e44f0f477b312622455cb', 'de0e344a655966aca556682e21b93ca6']:
			# 	self.hashfound = True
			# 	pass

			ret = self.multicore_run_all(Code(item))

			if ret.type == Block.type_no_error:
				try:
					eval = self.eval_code()

					if hash_prev_items + self.hash_code(item):
						prior += PRIOR_BOOST_IN_SNIPPET

					items.append(item)
					priors.append(prior)
					evals.append(eval)

					if len(items) >= max_moves:
						break

				except LookupError:
					pass

		if len(items) == 0:
			return []

		rewards = self.predict_rewards(np.stack(evals, axis = 0)).tolist()

		return list(zip(items, priors, rewards, evals))
