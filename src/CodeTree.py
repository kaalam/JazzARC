# Jazz (c) 2018-2026 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses

import hashlib
import math

from operator import itemgetter

from Bebop		import Bebop
from Block		import Block
from Code		import Code
from CodeBase	import CodeBase
from CodeEval	import CodeEval
from SearchConf	import WEIGHT_PRIOR_BY_FORM


class CodeTree(CodeEval):
	"""
	This contains dictionaries of primary and secondary structure code items.

	Once this object is built, its methods are not called, except those listed in this docstring, only its properties are used
	by CodeGen to identify possible code items (secondary structure groups of opcodes) and return them as candidates.

	Definitions:
	------------

	allele:		 Alleles are two code items that are identical in function opcodes and data types but differ in data opcode values.
	code item:	 A sequence of opcodes (cut from a snippet by build_codeitems_from_codebase). Its result (pic) can be evaluated.
	form:		 A tuple of arg_types/ret_type or data_type taken from a code item. Items with the same form are isomorphisms.
	variant:	 A tuple of bop functions and data types (not values) that defines all alleles for the same code item.
	isomorphism: Two code items with the same form.
	prior:		 The number of observations (n_item, n_form) weighted and linearly scaled to {0..1} == {min, max}

	Methods that will still be called after this object's construction:
	-------------------------------------------------------------------

	item_form(item)			- Returns the form to which an item belongs as a tuple (keys for dictionaries)
	item_variant(item)		- Returns the variant to which an item belongs as a tuple (keys for dictionaries)
	stack_use(npic, depth)	- Returns a single key adjusting npic by depth and depth only if bigger than npic, merging both.

	Properties computed during construction and kept unaltered in this object:
	--------------------------------------------------------------------------

	Primary structure (opcodes):

		self.opcode_by_arg_type - A tree of dictionaries (keys are types) organizing all the opcodes arg_types. All types other than
								  Block.type_nothing contain more dictionaries for more argument types. Block.type_nothing indicates
								  a leaf and holds a list of all opcodes match that specific argument type path.
		self.opcode_by_ret_type - A dictionary of lists of all the opcodes returning that type.

	Secondary structure (items):

		self.variant_by_stackuse	 Returns a list of variants by stack use as returned by stack_use()
		self.alleles_by_variant		 Returns a list of alleles by variant
		self.prior_by_codeitem		 Returns the prior value for the exact code item (allele)
		self.item_prior_by_stackuse	 Ready for CodeGen, returns (decreasing) sorted (item, prior) tuples to get everything in one place.
		self.in_snippets			 This is a set (all others are dictionaries) it contains the transitions from the previous (all items
									 concatenated in the snippet) and the item to be added as hash_code() values.

	One instance of MCTS is all you need since it inherits <- CodeGen <- <this> <- CodeEval <- Field <- Bebop.
	"""

	def __init__(self, code_base_fn, abstract_relation):
		CodeEval.__init__(self, abstract_relation)

		if code_base_fn is None:
			return

		self.cb = CodeBase(code_base_fn, abstract_relation)

		self.build_opcodes_by_types()
		self.build_codeitems_from_codebase()
		self.build_isomorphisms()
		self.aggregate_priors()


	def hash_code(self, code_item):
		"""
		Returns a hash for a code item (same decompile == same hash) for debugging purposes.
		"""
		source = self.decompile(code_item, pretty=False)
		s = ''.join(source)

		return hashlib.md5(bytes(s, encoding = 'utf-8')).hexdigest()


	@staticmethod
	def item_form(code_item):
		"""
		Returns the form defining all isomorphisms of an item.
		"""
		form = ()
		for op in code_item:
			if op.type == Block.type_function:
				form = form + (tuple(op.arg_types) + (op.ret_type,),)
			else:
				form = form + (op.type,)

		return form


	@staticmethod
	def item_variant(code_item):
		"""
		Returns the variant defining all alleles of an item.
		"""
		variant = ()
		for op in code_item:
			if op.type == Block.type_function:
				variant = variant + (op,)
			else:
				variant = variant + (op.type,)

		return variant


	@staticmethod
	def stack_use(npic, depth):
		"""
		Returns a signature of stack use to be used as a dictionary key. It also adjusts npic to the depth really accessed and makes
		depth zero when not deeper than npic.
		"""
		npic = min(npic, depth)
		if depth <= npic:
			depth = 0

		return int(1000*npic + depth)


	@staticmethod
	def stack_npic_depth(use):
		"""
		This is the inverse of stack_use(npic, depth). It returns npic, depth based on stack_use
		"""
		return int(use/1000), int(use % 1000)			# use = int(1000*npic + depth)


	def build_opcodes_by_types(self):
		"""
		Builds the dictionaries of all opcodes by argument and return types. The reference is this classe's description and the
		dictionaries built are: self.opcode_by_arg_type and self.opcode_by_ret_type
		"""
		self.opcode_by_arg_type = {}
		self.opcode_by_ret_type = {}

		for bop_fun in self.opcode.values():
			for i, arg_type in enumerate(bop_fun.arg_types):
				if i == 0:
					if arg_type not in self.opcode_by_arg_type:
						self.opcode_by_arg_type[arg_type] = {}
					ret_dic = self.opcode_by_arg_type[arg_type]
				else:
					if arg_type not in ret_dic:
						ret_dic[arg_type] = {}
					ret_dic = ret_dic[arg_type]

			if Block.type_nothing not in ret_dic:
				ret_dic[Block.type_nothing] = []

			ret_dic[Block.type_nothing].append(bop_fun)

			if bop_fun.ret_type not in self.opcode_by_ret_type:
				self.opcode_by_ret_type[bop_fun.ret_type] = []

			self.opcode_by_ret_type[bop_fun.ret_type].append(bop_fun)


	def push_item(self, code_item, npic, depth, prior):
		"""
		Add a newly detected code item to the dictionaries:

		self.variant_by_stackuse = {}
		self.alleles_by_variant	 = {}
		self.prior_by_codeitem	 = {}	This just counts the number of times seen and will be finally adjusted by aggregate_priors()
		"""
		use		= self.stack_use(npic, depth)
		variant = self.item_variant(code_item)

		if use not in self.variant_by_stackuse:
			self.variant_by_stackuse[use] = []

		if variant not in self.variant_by_stackuse[use]:
			self.variant_by_stackuse[use].append(variant)

		if variant not in self.alleles_by_variant:
			self.alleles_by_variant[variant] = []

		if code_item not in self.alleles_by_variant[variant]:
			self.alleles_by_variant[variant].append(code_item)

		if code_item not in self.prior_by_codeitem:
			self.prior_by_codeitem[code_item]  = prior
		else:
			self.prior_by_codeitem[code_item] += prior


	def build_codeitems_from_codebase(self):
		"""
		Identifies all codeitems in the codebase by running opcode by opcode checking picture depth and stack depth.

		1. Initializes dictionaries
		2. Uses rule "if l_stack_npic > 0 and opcode not in sto and last_op not in get_or_swap and last_op.type == Block.type_function:"
		   to break snippets into items.
		3. Calls push_item() on each new item to add it to the dictionaries.
		"""
		self.variant_by_stackuse = {}
		self.alleles_by_variant	 = {}
		self.prior_by_codeitem	 = {}
		self.in_snippets		 = set()

		self.demo_q = []
		self.demo_a = []

		get_or_swap = set([Bebop.bebop_get_question, Bebop.bebop_get_a, Bebop.bebop_get_b, Bebop.bebop_get_c, Bebop.bebop_get_d,
						   Bebop.bebop_get_e, Bebop.bebop_swap_top2, Bebop.bebop_swap_top3])

		sto = set([Bebop.bebop_sto_a, Bebop.bebop_sto_b, Bebop.bebop_sto_c, Bebop.bebop_sto_d, Bebop.bebop_sto_e])

		for code, _, name, question in self.cb:
			self.question = [question]

			self.multicore_clear()

			stack = self.multicore_state['stacks'][0]

			o_stack_npic = 0		# Beginning of item
			o_stack_size = 0
			o_min_height = 0

			l_stack_npic = 0		# Previous opcode
			l_stack_size = 0

			prev_items = ()

			item	= ()
			last_op = None

			for opcode in code.data:
				stack_height = len(stack)
				if opcode.type == Block.type_function:
					for typ in opcode.arg_types:
						if typ != Block.type_core:
							stack_height -= 1

				ret = self.multicore_run_all(Code((opcode,)), ignore_ret_type=True)

				assert ret.type == Block.type_no_error

				stack_npic = 0
				for blk in reversed(stack):
					if blk.type != Block.type_picture:
						break
					stack_npic += 1

				if l_stack_npic > 0 and opcode not in sto and last_op not in get_or_swap and last_op.type == Block.type_function:
					self.push_item(item, o_stack_npic, max(0, o_stack_size - o_min_height), 1)

					self.in_snippets.add(self.hash_code(prev_items) + self.hash_code(item))

					prev_items = prev_items + item

					item = ()

					o_stack_npic = l_stack_npic
					o_stack_size = l_stack_size
					o_min_height = l_stack_size

				item = item + (opcode,)
				o_min_height = min(o_min_height, stack_height)

				l_stack_npic = stack_npic
				l_stack_size = len(stack)
				last_op		 = opcode

			self.push_item(item, o_stack_npic, max(0, o_stack_size - o_min_height), 1)

			self.in_snippets.add(self.hash_code(prev_items) + self.hash_code(item))


	def isomorphisms(self, code_item):
		"""
		Creates a list of candidate isomorphisms by replacing each target by a random opcode of the same types.
		"""
		ret_type = set([Block.type_picture, Block.type_pictures, Block.type_pattern])

		ret = []

		l_code = list(code_item)

		for i, opcode in enumerate(code_item):
			if opcode.type == Block.type_function and opcode.ret_type in ret_type and Block.type_core not in opcode.arg_types:
				alt = [op for op in self.opcode_by_ret_type[opcode.ret_type] if op.arg_types == opcode.arg_types and op != opcode]

				for al in alt:
					iso = l_code[:i]
					iso.append(al)
					iso.extend(l_code[i + 1:])

					ret.append(tuple(iso))

		return ret


	def build_isomorphisms(self):
		"""
		Runs after build_codeitems_from_codebase() and extends the lists of most popular items by adding some isomorphisms replacing
		just one function at the time (not using core and returning either pic of pics). Isomorphisms are stored int the same
		dictionaries just like regular items but start with a different prior.

		Isomorphisms start with self.prior_by_codeitem = 0, but their final prior will be >0 since it uses prior_by_form and forms have
		isomorphisms with prior value > 0.
		"""
		new_iso	  = []
		new_npic  = []
		new_depth = []

		for use, variants in self.variant_by_stackuse.items():
			npic, depth = self.stack_npic_depth(use)
			for variant in variants:
				for code_item in self.alleles_by_variant[variant]:
					if self.prior_by_codeitem[code_item] > 1:
						for iso in self.isomorphisms(code_item):
							if iso not in self.prior_by_codeitem:
								new_iso.append(iso)
								new_npic.append(npic)
								new_depth.append(depth)

								self.prior_by_codeitem[iso] = 0

		for iso, npic, depth in zip(new_iso, new_npic, new_depth):
			self.push_item(iso, npic, depth, 0)


	def aggregate_priors(self):
		"""
		Normalizes the values in self.prior_by_codeitem to [0, 1], builds prior_by_form to assign priors to isomorphisms and merges
		all using WEIGHT_PRIOR_BY_FORM. Finally, puts it all in one place: self.item_prior_by_stackuse
		"""
		# Transform all values by log(1+x) to narrow the gap from 1s to 15s
		for item, prior in self.prior_by_codeitem.items():
			self.prior_by_codeitem[item] = math.log1p(prior)

		# Compute max, assert min
		min_prior =  9e9
		max_prior = -9e9

		for prior in self.prior_by_codeitem.values():
			min_prior = min(min_prior, prior)
			max_prior = max(max_prior, prior)

		assert min_prior == 0 and max_prior > 0

		# Scale all to [0..1]
		for item, prior in self.prior_by_codeitem.items():
			self.prior_by_codeitem[item] = prior/max_prior

		# Compute the average prior value over all isomorphisms within a form
		prior_by_form = {}
		n_by_form	  = {}
		for item, prior in self.prior_by_codeitem.items():
			form = self.item_form(item)

			if form in n_by_form:
				prior_by_form[form] += prior
				n_by_form[form]		+= 1
			else:
				prior_by_form[form]  = prior
				n_by_form[form]		 = 1

		for form, prior in prior_by_form.items():
			prior_by_form[form] = prior/n_by_form[form]

		# uses WEIGHT_PRIOR_BY_FORM as. prior = (1 - WEIGHT_PRIOR_BY_FORM)*prior + WEIGHT_PRIOR_BY_FORM*prior_by_form[form]
		w_prior = 1 - WEIGHT_PRIOR_BY_FORM
		for item, prior in self.prior_by_codeitem.items():
			form = self.item_form(item)
			self.prior_by_codeitem[item] = w_prior*prior + WEIGHT_PRIOR_BY_FORM*prior_by_form[form]

		# Builds self.item_prior_by_stackuse, ready for CodeGen, (decreasing) sorted (item, prior) tuples to get everything in one place.
		self.item_prior_by_stackuse = {}
		for use, variants in self.variant_by_stackuse.items():
			items  = []
			priors = []
			for variant in variants:
				for item in self.alleles_by_variant[variant]:
					items.append(item)
					priors.append(self.prior_by_codeitem[item])

			self.item_prior_by_stackuse[use] = sorted(zip(items, priors), key=itemgetter(1), reverse=True)
