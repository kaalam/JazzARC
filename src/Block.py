# Jazz (c) 2018-2025 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses

import numpy as np


class Block:
	"""
	This is the common ancestor of all classes forming the language (data and functions).

	It works as a class to be instanced via new_*() E.g., `Block.new_picture(list_of_list=[[30, 60, 90], [8, 0, 1]])`

	W a r n i n g : Don't use the new_picture() method in any other way that as an auxiliary constructor!! x = Block.new_picture()
	--------------- It contains a mutable argument with a default value. In Python, when passing a mutable value as a default argument
					in a function, the default argument is mutated anytime that value is mutated.
					(https://docs.python-guide.org/writing/gotchas/)
	"""

	def __init__(self, data, type):
		self.data = data
		self.type = type


	@classmethod
	def new_integer(bl_class, item):
		"""
		Creates a block containing an integer.
		"""
		return bl_class(item, Block.type_integer)


	@classmethod
	def new_int_pair(bl_class, tuple_of_two_int):
		"""
		Creates a block containing a tuple of two integers.
		"""
		return bl_class(tuple_of_two_int, Block.type_int_pair)


	@classmethod
	def new_error(bl_class, error_msg):
		"""
		Creates a block containing an error message. This mechanism is how bop_function() decorated functions return errors.
		"""
		return bl_class(error_msg, Block.type_error)


	@classmethod
	def new_function(bl_class, item):
		"""
		Creates a block containing a function. This is encapsulated by the class Function which also takes care of
		argument and return types.
		"""
		return bl_class(item, Block.type_function)


	@classmethod
	def new_NESW(bl_class, tuple_of_4_nesw):
		"""
		Creates a block containing a tuple of 4 integer in directions N, E, S, W.
		"""
		return bl_class(tuple_of_4_nesw, Block.type_NESW)


	@classmethod
	def new_pattern(bl_class, pic, mask):
		"""
		Creates a block containing a tuple of a picture and a mask.
		"""
		return bl_class((pic, mask), Block.type_pattern)


	@classmethod
	def new_picture(bl_class, np_arr=None, list_of_list=None):
		"""
		Creates a block containing a picture. You can give either a numpy array or a list of list.
		"""
		if np_arr is None:
			return bl_class(np.array(list_of_list, dtype=np.int32), Block.type_picture)
		else:
			return bl_class(np_arr, Block.type_picture)


	@classmethod
	def new_pictures(bl_class, tuple_of_np_array):
		"""
		Creates a block containing a tuple of pictures (np.array of int.32).
		"""
		return bl_class(tuple_of_np_array, Block.type_pictures)


	@classmethod
	def new_vector(bl_class, vector_of_int):
		"""
		Creates a block containing a list of integers stored as Block.type_vector.
		"""
		return bl_class(vector_of_int, Block.type_vector)


# 	Block.type values
# 	-----------------

# 	These methods are never called, they are just memory addresses to check types.

# 	E.g.,
# 		b = Block.new_function(..)
# 		assert b.type == Block.type_function
# _________________________________________________________________________________________________________________________________

	@staticmethod
	def type_integer():
		"""
		This type is one single integer. E.g., for functions accepting a background color (other than 0), this is the argument type.
		For a pair of colors, as in recolor, use int_pair. When written in source code, it must be given as a tuple of one element
		to make the compiler recognize it.

		Note: Tuples are defined by length: type_integer, type_int_pair and type_NESW assigned by the compiler based on length.

		The source code form is: (1) (A python integer in brackets. It is NOT a tuple.)
		"""
		return 'integer'


	@staticmethod
	def type_int_pair():
		"""
		This type is a tuple of two integers. E.g., what a recolor function would expect.

		Note: Tuples are defined by length: type_integer, type_int_pair and type_NESW assigned by the compiler based on length.

		The source code form is: (1, 2) (A python tuple of 2 integers.)
		"""
		return 'int_pair'


	@staticmethod
	def type_core():
		"""
		This type is used for arguments only. All functions are Block.type_function, but core functions, e.g.,
		the functions needing to use the stack directly, use this type as an argument type and they get the core
		object passed as an argument.

		This type has no source code form.
		"""
		return 'core'


	@staticmethod
	def type_error():
		"""
		This type is used for return values only. Any function can return errors by returning a Block.new_error('This went wrong')
		instead of returning the expected result. This will force the Core running the program to hcf().

		This type has no source code form.
		"""
		return 'error'


	@staticmethod
	def type_function():
		"""
		Functions returned by Block.new_function()

		The source code form is just the name of the function with no extra characters anywhere.
		"""
		return 'function'


	@staticmethod
	def type_NESW():
		"""
		A tuple of four integers. This describes moves, extensions/reductions and similar things.
		It defines movement in dim 4, North, East, South, West (NESW)

		Note: Tuples are defined by length: type_integer, type_int_pair and type_NESW assigned by the compiler based on length.

		The source code form is: (1, 2, 3, 4) (A python tuple of 4 integers.)
		"""
		return 'NESW'


	@staticmethod
	def type_no_error():
		"""
		This type is used for Block descendants only. It is the opposite of an error. E.g., This makes it easy to verify if a
		compilation was successful. Field(compile) returns a Code() instance (a Block descendant), but on error it returns a
		Block.new_error(message), so checking ret.type is enough. There is no need to guess descendants and create a bunch of
		them: type_code, type_source, type_bond, type_pizza, ... Same type for all, this one.

		This type has no source code form.
		"""
		return 'no_error'


	@staticmethod
	def type_nothing():
		"""
		This type is used for return values only. The only functions returning nothing are core functions.
		E.g., core_dup_stack_top()

		This type has no source code form.
		"""
		return 'nothing'


	@staticmethod
	def type_pattern():
		"""
		A pattern is a tuple of a picture and a Numpy boolean selector, a mask, with the same shape selecting pixels.
		The logic must be equivalent to using Numpy's `pic[mask]`.

		Nonselected pixels must have no effect in the returned value in every function accepting this type!!

		This type has no source code form: You cannot create constants of this type, but some functions return it and some accept it.
		You can also create a constant picture and convert it to a pattern using pic_all_as_pat.
		"""
		return 'pattern'


	@staticmethod
	def type_picture():
		"""
		A Numpy array of rank 2 and type np.int32.

		The source code form is: [[1, 2], [3, 4]] (A python list of lists that an np.array of np.int32 would like.)
		"""
		return 'picture'


	@staticmethod
	def type_pictures():
		"""
		A tuple of pictures.

		Internally it is a python tuple of Numpy arrays of rank 2 and type np.int32.

		This type has no source code form: You cannot create constants of this type, but some functions return it and some accept it.
		"""
		return 'pictures'


	@staticmethod
	def type_vector():
		"""
		This type is a vector of integers of any length. This is typically used in functions that do "smart" things
		like recoloring by areas, picking a number of colors, etc. See e.g., 7c008303.json

		The source code for this is [1, 2, 3] like a python list and it is stored as a python list. Not being a tuple
		avoids mixing it with type_integer and friends. Not starting with [[ avoids treating it as a picture.
		"""
		return 'vector'
