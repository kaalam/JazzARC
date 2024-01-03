# Jazz (c) 2018-2024 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses

import re

from Block	  import Block
from Function import bop_function


class Bebop():
	"""
	This root class completes the basics of language (i.e., the non DSL parts).

	Below Bebop:

		- Block manages the creation and typing of the objects (callable and not callable) used in the language.
		- Function creates the callable blocks by providing a class and a decorater.
		- Code creates blocks of executable snippets.
		- Core runs Code objects using an internal stack.

	The Bebop class:

		- Manages dictionaries of callable blocks by name to support compilation.
		- Stores the core functions of the language.

	Above Bebop:

		- Source provides storage for source code snippets.
		- Bond defines the only Relation in this PoC between two Block descendants ori and dst.
		- Example inherits Bond to make it link Questions to Answers (with an extra is_test argument).
		- Field provides the mechanism to compile code for a Bond descendant.
		- Field provides support for running the compiled code.

	On the main level

		- Search provides an API to BaseCodes, MCTS, Structure, Fwdprop and Backprop which use all the other classes.
	"""

	def __init__(self):
		self.opcode = {}
		self.use(Bebop)


	def use(self, abstract_class):
		"""
		Updates the self.opcode dictionary with all functions starting with the (lowercase) name of the class.
		"""
		prefix = abstract_class.__name__.lower()
		rex	   = re.compile('^' + prefix + '_([A-Za-z0-9_]+)$')

		for method in filter(rex.match, dir(abstract_class)):
			kk = rex.sub('\\1', method)
			vv = getattr(abstract_class, method)

			if kk in self.opcode:
				raise KeyError

			self.opcode[kk] = vv


	@bop_function(arg_types=[Block.type_core], ret_type=Block.type_nothing)
	def bebop_get_a(core):
		"""
		Pushes the content of core register 'a' or hcf() if it is None.
		"""
		aa = core.register['a']

		if aa is None:
			return core.hcf('get_a() empty register')

		core.stack.append(aa)


	@bop_function(arg_types=[Block.type_core], ret_type=Block.type_nothing)
	def bebop_get_answer(core):
		"""
		Pushes the content of core register 'answer' or hcf() if it is None.
		"""
		aa = core.register['answer']

		if aa is None:
			return core.hcf('get_answer() empty register')

		core.stack.append(aa)


	@bop_function(arg_types=[Block.type_core], ret_type=Block.type_nothing)
	def bebop_get_b(core):
		"""
		Pushes the content of core register 'b' or hcf() if it is None.
		"""
		bb = core.register['b']

		if bb is None:
			return core.hcf('get_b() empty register')

		core.stack.append(bb)


	@bop_function(arg_types=[Block.type_core], ret_type=Block.type_nothing)
	def bebop_get_c(core):
		"""
		Pushes the content of core register 'c' or hcf() if it is None.
		"""
		cc = core.register['c']

		if cc is None:
			return core.hcf('get_c() empty register')

		core.stack.append(cc)


	@bop_function(arg_types=[Block.type_core], ret_type=Block.type_nothing)
	def bebop_get_d(core):
		"""
		Pushes the content of core register 'd' or hcf() if it is None.
		"""
		dd = core.register['d']

		if dd is None:
			return core.hcf('get_d() empty register')

		core.stack.append(dd)


	@bop_function(arg_types=[Block.type_core], ret_type=Block.type_nothing)
	def bebop_get_e(core):
		"""
		Pushes the content of core register 'e' or hcf() if it is None.
		"""
		ee = core.register['e']

		if ee is None:
			return core.hcf('get_e() empty register')

		core.stack.append(ee)


	@bop_function(arg_types=[Block.type_core], ret_type=Block.type_nothing)
	def bebop_get_question(core):
		"""
		Pushes the content of core register 'question' or hcf() if it is None.
		"""
		qq = core.register['question']

		if qq is None:
			return core.hcf('get_question() empty register')

		core.stack.append(qq)


	@bop_function(arg_types=[Block.type_core, Block.type_pictures], ret_type=Block.type_nothing)
	def bebop_pics_as_2pic(core, pics):
		"""
		Explodes the tuple of just two pic (hcf() on any error) pushing them as pictures.
		"""
		if pics.type != Block.type_pictures or len(pics.data) != 2:
			return core.hcf('pics_as_2pic() tuple of two pictures expected')

		for pic in pics.data:
			core.stack.append(Block.new_picture(pic))


	@bop_function(arg_types=[Block.type_core, Block.type_pictures], ret_type=Block.type_nothing)
	def bebop_pics_as_3pic(core, pics):
		"""
		Explodes the tuple of just three pic (hcf() on any error) pushing them as pictures.
		"""
		if pics.type != Block.type_pictures or len(pics.data) != 3:
			return core.hcf('pics_as_3pic() tuple of three pictures expected')

		for pic in pics.data:
			core.stack.append(Block.new_picture(pic))


	@bop_function(arg_types=[Block.type_core, Block.type_pictures], ret_type=Block.type_nothing)
	def bebop_pics_as_pic(core, pics):
		"""
		Explodes the tuple of just one pic (hcf() on any error) pushing it as a picture.
		"""
		if pics.type != Block.type_pictures or len(pics.data) != 1:
			return core.hcf('pics_as_pic() tuple of one picture expected')

		for pic in pics.data:
			core.stack.append(Block.new_picture(pic))


	@bop_function(arg_types=[Block.type_core], ret_type=Block.type_nothing)
	def bebop_sto_a(core):
		"""
		Copies the tos and stores it in the core register 'a' or hcf() if stack is empty.
		"""
		if not core.stack:
			return core.hcf('sto_a() empty stack')

		core.register['a'] = core.stack[-1]


	@bop_function(arg_types=[Block.type_core], ret_type=Block.type_nothing)
	def bebop_sto_b(core):
		"""
		Copies the tos and stores it in the core register 'b' or hcf() if stack is empty.
		"""
		if not core.stack:
			return core.hcf('sto_b() empty stack')

		core.register['b'] = core.stack[-1]


	@bop_function(arg_types=[Block.type_core], ret_type=Block.type_nothing)
	def bebop_sto_c(core):
		"""
		Copies the tos and stores it in the core register 'c' or hcf() if stack is empty.
		"""
		if not core.stack:
			return core.hcf('sto_c() empty stack')

		core.register['c'] = core.stack[-1]


	@bop_function(arg_types=[Block.type_core], ret_type=Block.type_nothing)
	def bebop_sto_d(core):
		"""
		Copies the tos and stores it in the core register 'd' or hcf() if stack is empty.
		"""
		if not core.stack:
			return core.hcf('sto_d() empty stack')

		core.register['d'] = core.stack[-1]


	@bop_function(arg_types=[Block.type_core], ret_type=Block.type_nothing)
	def bebop_sto_e(core):
		"""
		Copies the tos and stores it in the core register 'e' or hcf() if stack is empty.
		"""
		if not core.stack:
			return core.hcf('sto_e() empty stack')

		core.register['e'] = core.stack[-1]


	@bop_function(arg_types=[Block.type_core], ret_type=Block.type_nothing)
	def bebop_swap_top2(core):
		"""
		Swaps the two blocks at the tos.
		"""
		try:
			core.stack[-1], core.stack[-2] = core.stack[-2], core.stack[-1]
		except IndexError:
			core.hcf('swap_top2() with less than two')


	@bop_function(arg_types=[Block.type_core], ret_type=Block.type_nothing)
	def bebop_swap_top3(core):
		"""
		Swaps the three blocks at the tos.
		"""
		try:
			core.stack[-1], core.stack[-3] = core.stack[-3], core.stack[-1]
		except IndexError:
			core.hcf('swap_top3() with less than three')
