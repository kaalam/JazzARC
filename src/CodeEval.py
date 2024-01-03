# Jazz (c) 2018-2024 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses

import copy

from collections import deque

import numpy as np

from Block		import Block
from Core		import Core
from Field		import Field
from SearchConf	import EVAL_FULL_MATCH
from SearchConf	import LENGTH_CODE_EVAL
from SearchConf	import EVAL_WRONG_SHAPE
from SearchConf	import EVAL_MAX_PIC_SIZE
from SearchConf	import IDX_PIC_REACH_MIN
from SearchConf	import IDX_PIC_BETTER_MIN
from SearchConf	import IDX_PIC_WORSE_MIN
from SearchConf	import IDX_PAT_REACH_MIN
from SearchConf	import IDX_PAT_BETTER_MIN
from SearchConf	import IDX_PAT_WORSE_MIN
from SearchConf	import IDX_PIC_REACH_MEAN
from SearchConf	import IDX_PIC_BETTER_MEAN
from SearchConf	import IDX_PIC_WORSE_MEAN
from SearchConf	import IDX_PAT_REACH_MEAN
from SearchConf	import IDX_PAT_BETTER_MEAN
from SearchConf	import IDX_PAT_WORSE_MEAN
from SearchConf	import IDX_PIC_REACH_MAX
from SearchConf	import IDX_PIC_BETTER_MAX
from SearchConf	import IDX_PIC_WORSE_MAX
from SearchConf	import IDX_PAT_REACH_MAX
from SearchConf	import IDX_PAT_BETTER_MAX
from SearchConf	import IDX_PAT_WORSE_MAX


class CodeEval(Field):
	"""
	This class contains:

		Parallel execution unit multicore:
		----------------------------------

		multicore_clear()			   : Sets everything: storing self.demo_q/question, self.demo_a in registers and clearing stacks.
		multicore_run_all(code)		   : Runs a code item (possibly after many others without clearing).
		multicore_copy_state(state)	   : Makes a safe (== just 2 levels deep) copy of the state (used after tree walk, before eval).
		multicore_set_state(new_state) : Recovers the state to run the final part again evaluating a different child of the same node.

		Evaluation function:
		--------------------

		**simple_eval**(code, target) is a function returning:

			VALUE_FULL_MATCH	(positive value far from 0)
			num_ok/num_total	(0..1 for normal)
			VALUE_WRONG_SHAPE	(negative value close 0)

		**Evaluation** functions are used when nodes are expanded and replace the stochastic playout in the forward walk.

		Evaluation results are hierarchical of shape (2, 3, 3) over

			- pic or pattern			(x2)	either full color or just (not black)
			- absolute or incremental	(x3)	(reach, better, worse)
												**reach**  simple_eval(p_t, p_w)
												**better** (# pixels (in 0-1) that were wrong in p_t-1 and are right in p_t)
												**worse**  (# pixels (in 0-1) that were right in p_t-1 and are wrong in p_t)
			- over all examples			(x3)	(min, mean, max) for any problem regardless of the number of examples

	One instance of MCTS is all you need since it inherits <- CodeGen <- CodeTree <- <this> <- Field <- Bebop.
	"""

	def __init__(self, abstract_relation):
		Field.__init__(self, abstract_relation)

		self.multicore_state = None


	def multicore_clear(self):
		"""
		Gets everything ready storing self.demo_q, self.demo_a in the registers and clearing the stacks.
		"""
		stacks	   = []
		registers  = []
		pic_lists  = []

		for q, a in zip(self.demo_q, self.demo_a):
			stacks.append(deque())
			reg = {'a' : None, 'b' : None, 'c' : None, 'd' : None, 'e' : None, 'question' : q, 'answer' : a}
			registers.append(reg)
			pic_lists.append([])

		for q in self.question:
			stacks.append(deque())
			reg = {'a' : None, 'b' : None, 'c' : None, 'd' : None, 'e' : None, 'question' : q, 'answer' : None}
			registers.append(reg)
			pic_lists.append([])

		del self.multicore_state

		self.multicore_state = {
			'stacks'	 : stacks,
			'registers'	 : registers,
			'pic_lists'	 : pic_lists}


	def multicore_run_all(self, code, ignore_ret_type=False):
		"""
		Runs a code item (possibly after many others without clearing).
		"""
		stacks	   = self.multicore_state['stacks']
		registers  = self.multicore_state['registers']
		pic_lists  = self.multicore_state['pic_lists']

		for stack, register, pic_list in zip(stacks, registers, pic_lists):
			core = Core(code, stack, register)

			try:
				*_, ret = core

			except Exception:
				return Block.new_error('Try/catch caught an exception')

			if ret.type == Block.type_error or not core.all_right:
				return ret

			if ret.type == Block.type_picture:
				pic_list.append(ret)
			elif not ignore_ret_type:
				return Block.new_error('Code item does not return a picture')

			stack	 = core.stack
			register = core.register

		return Block(None, Block.type_no_error)


	def multicore_copy_state(self, state):
		"""
		Stores the state by copying everything (after tree walk, before eval)
		"""
		stacks	   = state['stacks']
		registers  = state['registers']
		pic_lists  = state['pic_lists']

		copy_stacks		= []
		copy_registers  = []
		copy_pic_lists  = []

		for item in stacks:
			copy_stacks.append(copy.copy(item))

		for item in registers:
			copy_registers.append(copy.copy(item))

		for item in pic_lists:
			copy_pic_lists.append(copy.copy(item))

		copy_state = {
			'stacks'	 : copy_stacks,
			'registers'	 : copy_registers,
			'pic_lists'	 : copy_pic_lists}

		return copy_state


	def multicore_set_state(self, new_state):
		"""
		Recovers the state to run the final part again evaluating a different child of the same node.
		"""
		del self.multicore_state

		self.multicore_state = self.multicore_copy_state(new_state)


	def eval_code(self):
		"""
		Evaluates a move as described above. It uses the multicore state and requires at least one picture in the pìc_list for all
		cores with an answer.

		Returns a numpy array of 18 (LENGTH_CODE_EVAL) values indexed by: IDX_PIC_REACH_MIN..IDX_PAT_WORSE_MAX
		"""
		eval = np.repeat((9e9, 0, -9e9), LENGTH_CODE_EVAL/3)
		N_ex = 0
		for reg, pic in zip(self.multicore_state['registers'], self.multicore_state['pic_lists']):
			if reg['answer'] is None:
				break

			pic_t3 = reg['answer'].data

			if len(pic) >= 2:
				pic_t1 = pic[-2].data
				pic_t2 = pic[-1].data
			elif len(pic) == 1:
				pic_t1 = reg['question'].data
				pic_t2 = pic[0].data
			else:
				raise LookupError

			if pic_t2.shape != pic_t3.shape:
				if pic_t2.shape[0] > EVAL_MAX_PIC_SIZE or pic_t2.shape[1] > EVAL_MAX_PIC_SIZE:
					raise LookupError

				pic_reach  = EVAL_WRONG_SHAPE
				pic_better = EVAL_WRONG_SHAPE
				pic_worse  = EVAL_WRONG_SHAPE
				pat_reach  = EVAL_WRONG_SHAPE
				pat_better = EVAL_WRONG_SHAPE
				pat_worse  = EVAL_WRONG_SHAPE
			else:
				N_tot = pic_t2.shape[0]*pic_t2.shape[1]

				N_equ = int(0.5 + np.sum(pic_t2 == pic_t3))

				if N_equ == N_tot:
					pic_reach = EVAL_FULL_MATCH
				else:
					pic_reach = N_equ/N_tot

				pat_t2 = pic_t2 != 0
				pat_t3 = pic_t3 != 0

				N_equ = int(0.5 + np.sum(pat_t2 == pat_t3))

				if N_equ == N_tot:
					pat_reach = EVAL_FULL_MATCH
				else:
					pat_reach = N_equ/N_tot

				if pic_t1.shape != pic_t2.shape:
					pic_better = EVAL_WRONG_SHAPE
					pic_worse  = EVAL_WRONG_SHAPE
					pat_better = EVAL_WRONG_SHAPE
					pat_worse  = EVAL_WRONG_SHAPE
				else:
					was_ok = pic_t1 == pic_t3
					is_ok  = pic_t2 == pic_t3

					pic_better = np.sum(np.logical_and(is_ok,  np.logical_not(was_ok)))/N_tot
					pic_worse  = np.sum(np.logical_and(was_ok, np.logical_not(is_ok)))/N_tot

					pat_t1 = pic_t1 != 0

					was_ok = pat_t1 == pat_t3
					is_ok  = pat_t2 == pat_t3

					pat_better = np.sum(np.logical_and(is_ok,  np.logical_not(was_ok)))/N_tot
					pat_worse  = np.sum(np.logical_and(was_ok, np.logical_not(is_ok)))/N_tot

			eval[IDX_PIC_REACH_MIN]	   = min(eval[IDX_PIC_REACH_MIN],  pic_reach)
			eval[IDX_PIC_BETTER_MIN]   = min(eval[IDX_PIC_BETTER_MIN], pic_better)
			eval[IDX_PIC_WORSE_MIN]	   = min(eval[IDX_PIC_WORSE_MIN],  pic_worse)
			eval[IDX_PAT_REACH_MIN]	   = min(eval[IDX_PAT_REACH_MIN],  pat_reach)
			eval[IDX_PAT_BETTER_MIN]   = min(eval[IDX_PAT_BETTER_MIN], pat_better)
			eval[IDX_PAT_WORSE_MIN]	   = min(eval[IDX_PAT_WORSE_MIN],  pat_worse)
			eval[IDX_PIC_REACH_MEAN]  += pic_reach
			eval[IDX_PIC_BETTER_MEAN] += pic_better
			eval[IDX_PIC_WORSE_MEAN]  += pic_worse
			eval[IDX_PAT_REACH_MEAN]  += pat_reach
			eval[IDX_PAT_BETTER_MEAN] += pat_better
			eval[IDX_PAT_WORSE_MEAN]  += pat_worse
			eval[IDX_PIC_REACH_MAX]	   = max(eval[IDX_PIC_REACH_MAX],  pic_reach)
			eval[IDX_PIC_BETTER_MAX]   = max(eval[IDX_PIC_BETTER_MAX], pic_better)
			eval[IDX_PIC_WORSE_MAX]	   = max(eval[IDX_PIC_WORSE_MAX],  pic_worse)
			eval[IDX_PAT_REACH_MAX]	   = max(eval[IDX_PAT_REACH_MAX],  pat_reach)
			eval[IDX_PAT_BETTER_MAX]   = max(eval[IDX_PAT_BETTER_MAX], pat_better)
			eval[IDX_PAT_WORSE_MAX]	   = max(eval[IDX_PAT_WORSE_MAX],  pat_worse)

			N_ex += 1

		if N_ex == 0:
			raise LookupError

		eval[IDX_PIC_REACH_MEAN]  /= N_ex
		eval[IDX_PIC_BETTER_MEAN] /= N_ex
		eval[IDX_PIC_WORSE_MEAN]  /= N_ex
		eval[IDX_PAT_REACH_MEAN]  /= N_ex
		eval[IDX_PAT_BETTER_MEAN] /= N_ex
		eval[IDX_PAT_WORSE_MEAN]  /= N_ex

		return eval
