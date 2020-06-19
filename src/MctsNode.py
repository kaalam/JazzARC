# Jazz (c) 2018-2020 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses

import math

from operator import itemgetter

import numpy as np

from SearchConf import DIRICHLET_ALPHA
from SearchConf import EXPLORATION_FRAC
from SearchConf import UCB_C_BASE
from SearchConf import UCB_C_INIT
from SearchConf import UCB_DISCOUNT


class MctsNode():
	"""
	This class implements nodes in the tree used by MCTS.
	"""

	def __init__(self, code=None, prior=0, reward=None, parent=None):
		"""
		Creates a new node. Except for root, all arguments should be given.

		code   - A secondary structure snippet (as a tuple of opcodes) that can be evaluated.
		prior  - The prior for the code item in the CodeTree.
		reward - The CodeEval of the code (from room to node) converted into a reward by CodeGen.predict_rewards().
		parent - The parent (MctsNode) from which it will hang when created.
		"""
		# Tree structure
		self.parent	  = parent
		self.children = []

		# State
		self.code = code

		# UCB navigation
		self.prior	= prior
		self.reward	= 0 if reward is None else reward
		self.visits	= 0 if reward is None else 1

		if parent is not None:
			self.parent.children.append(self)


	def is_root(self):
		"""
		Returns if a node is root (used in backwards walks)
		"""
		return self.parent is None


	def is_leaf(self):
		"""
		Returns if a node is a leaf (used in forward walks)
		"""
		return self.children == []


	def ucb_score(self, child):
		"""
		Computes the UCB score of a child. This assumes self is the parent of the child as used in select_child()
		"""
		pb_c  = math.log((self.visits + UCB_C_BASE + 1)/UCB_C_BASE) + UCB_C_INIT
		pb_c *= math.sqrt(self.visits)/(child.visits + 1)

		prior_score = pb_c*child.prior
		value_score = UCB_DISCOUNT*(child.reward/(child.visits + 1))

		return prior_score + value_score


	def select_child(self):
		"""
		Selects the child with the highest UCB score.
		"""
		_, child = max(((self.ucb_score(child), child) for child in self.children), key=itemgetter(0))

		return child


	def add_exploration_noise(self):
		"""
		At the start of each search, we add dirichlet noise to the prior of the root to encourage the search to explore new actions.

		In the Deepmind pseudocode, they add this to the root (== self) node only. Forcing exploration for the first move only.
		"""
		N = len(self.children)

		noise = np.random.dirichlet([DIRICHLET_ALPHA]*N)

		for child, n in zip(self.children, noise):
			child.prior = child.prior*(1 - EXPLORATION_FRAC) + n*EXPLORATION_FRAC


	def print(self, indent='', is_last=None, ucb=0, recurse_depth=10, field=None):
		depth = len(indent)/2

		if is_last is None:
			if self.parent is None:
				is_last = True
			else:
				is_last = self.parent.children[-1] is self

		if depth > 0:
			print(indent + '|')

		if is_last:
			fi_line	= indent + '\\-'
			x_lines	= indent + '  >'
			y_lines	= indent + '  |'
			indent	= indent + '  '
		else:
			fi_line	= indent + '+-'
			x_lines	= indent + '| >'
			y_lines	= indent + '| |'
			indent	= indent + '| '

		values = (hex(id(self)), ucb, self.prior, self.reward, self.visits)
		line   = fi_line + '%14s  ucb: %6.3f  prior: %4.3f  reward: %6.3f  visits: %4i' % values

		print(line)

		if field is not None and self.code is not None:
			print(x_lines)
			txt = [x_lines + s for s in field.decompile(self.code)]
			for line in txt:
				print(line)

		if depth < recurse_depth:
			N	   = len(self.children)
			n_void = 0

			summarized = False

			gen = sorted(((child.visits, self.ucb_score(child), child) for child in self.children), key=itemgetter(0, 1), reverse=True)
			for i, (visits, ucb, child) in enumerate(gen):
				n_void = 0 if visits > 2 else n_void + 1
				if i == N - 1:
					child.print(indent=indent, is_last=True, ucb=ucb, recurse_depth=recurse_depth, field=field)
				else:
					if n_void <= 5:
						child.print(indent=indent, is_last=False, ucb=ucb, recurse_depth=recurse_depth, field=field)
					elif not summarized:
						print(y_lines)
						print(y_lines + '%20s. . . %i children in all . . .' % (' ', N))
						summarized = True
