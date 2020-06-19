# Jazz (c) 2018-2020 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses

import re

from Bebop import Bebop
from Block import Block
from Code  import Code
from Core  import Core


class Field(Bebop):
	"""
	This implements formal fields.

	Field provides the mechanism to compile and decompile code for a Bond descendant.
	Field provides support for running the compiled code.

	The language supported is in abstract classes returned by .function_classes() of the Block descendants.

	One Field instance (which is defined by the classes making the ori and dst of its Relation) can run code on an unlimited
	number of Bond descendants. (I.e., Building the Field is slow, but switching examples is fast.)

	Unlike its parent, Bebop, the constructor needs an abstract class defining the relation to learn the language.
	"""

	def __init__(self, abstract_relation):
		Bebop.__init__(self)

		self.from_kind = abstract_relation.from_kind()
		self.to_kind   = abstract_relation.to_kind()

		self.rex_tuple	 = re.compile('^\\(.+\\)$')
		self.rex_picture = re.compile('^\\[\\[.+\\]\\]$')
		self.rex_vector	 = re.compile('^\\[[^\\[]+\\]$')

		for abs_class in abstract_relation.function_classes():
			self.use(abs_class)


	def compile(self, source):
		"""
		Compiles a Bebop Source object into a Bebop Code object.
		"""
		code_tuple = ()
		for statement in source.data:
			if self.rex_tuple.match(statement):
				try:
					tup = eval(statement)
				except Exception:
					return Block.new_error('Malformed tuple: ' + statement)

				if len(tup) == 1:
					code_tuple = code_tuple + (Block.new_integer(tup[0]),)
				elif len(tup) == 2:
					code_tuple = code_tuple + (Block.new_int_pair(tup),)
				elif len(tup) == 4:
					code_tuple = code_tuple + (Block.new_NESW(tup),)
				else:
					return Block.new_error('Tuple must be (color, int_pair or nesw): ' + statement)

			elif self.rex_picture.match(statement):
				try:
					lol = eval(statement)
				except Exception:
					return Block.new_error('Malformed picture: ' + statement)

				code_tuple = code_tuple + (Block.new_picture(list_of_list=lol),)

			elif self.rex_vector.match(statement):
				try:
					vec = eval(statement)
				except Exception:
					return Block.new_error('Malformed vector: ' + statement)

				code_tuple = code_tuple + (Block.new_vector(vec),)

			else:
				if statement in self.opcode:
					code_tuple = code_tuple + (self.opcode[statement],)
				else:
					return Block.new_error('Unknown opcode: ' + statement)

		if len(code_tuple) == 0:
			return Block.new_error('Empty source')

		return Code(code_tuple)


	def decompile(self, code, pretty=True):
		"""
		Sources any piece of bop code (Function, Blocks of integer, intp, nesw, vectors an pictures), tuples of these of Code() objects.

		This utility can be used for debugging (pretty=True) as a human readable list of strings which cannot be compiled.
		or it can be used (pretty=False) for serializing code that can be compiled again int a Code() object.
		"""
		if not isinstance(code, tuple):
			if code.type == Block.type_no_error:
				code = code.data
			else:
				code = (code,)

		if pretty:
			ret = []
		else:
			ret = ()

		for blk in code:
			if pretty:
				if blk.type == Block.type_function:
					statement = [k for k, v in self.opcode.items() if v == blk]

					assert len(statement) == 1

					args = ', '.join(map(lambda x : x(), blk.arg_types))

					ret.append(' %39s : %s(%s)' % (statement[0], blk.ret_type(), args))
				elif blk.type == Block.type_picture:
					ret.append(' %39s' % str(blk.data.tolist()))
				else:
					ret.append(' %39s' % str(blk.data))
			else:
				if blk.type == Block.type_function:
					statement = [k for k, v in self.opcode.items() if v == blk]
					ret += (statement[0],)
				elif blk.type == Block.type_picture:
					ret += (str(blk.data.tolist()),)
				elif blk.type == Block.type_integer:
					ret += (str((blk.data,)),)
				else:
					ret += (str(blk.data),)

		return ret


	def run(self, code, relation=None, peek_answer=False):
		"""
		Runs code over the data of a relation instance.

		If relation argument is given, the relation is assumed to be a Block of type tuple with the first two elements being
		the from and to Kind instances and possibly more elements. The instances are pushed to the stack by this method.

		Optionally, an initial stack can be given.

		It returns the top of the stack after the last code block (or last executed block on error) is run. All possible errors
		force this to be a Block of type_error. No valid programs can return a type_error.
		"""

		core = Core(code)

		if relation is not None:
			question, answer, is_test = relation.data

			core.register['question'] = question

			if peek_answer or not is_test:
				core.register['answer']	= answer

		*_, ret = core

		return ret
