# Jazz (c) 2018-2020 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses

import numpy as np

from Block	  import Block
from Function import bop_function


class BopBack():
	"""
	This abstract class is the storage of @bop_function decorated Bebop functions for all the search-specific non-DSL code. (I.e., code
	with access to the solution).

	The functions in this class return, if implemented, things like:

	- If both pics have the same shape
	- If both pics are identical
	- The number of different pixels
	- The pixel difference as a pattern
	- "Reverse engineering" ideas to guess what could convert one in the other

	Check the doc of BopForward for naming convention.
	"""

	@bop_function(arg_types=[Block.type_core], ret_type=Block.type_integer)
	def bopback_tests_verify_answer(core):
		"""
		Returns if the tos is a picture and is identical to the one stored in the core register['answer'].
		"""
		answer = core.register['answer']

		if answer is None or not core.stack:
			return Block.new_integer(0)

		pic = core.stack[-1]

		if pic.type != Block.type_picture:
			return Block.new_integer(0)

		ret = np.array_equal(pic.data, answer.data)*1

		return Block.new_integer(ret)


	# @bop_function(arg_types=[Block.type_picture, Block.type_picture], ret_type=Block.type_integer)
	# def bopback_2pic_same_shape_as_int(pic1, pic2):
	# 	"""
	# 	Returns if two pics have the same shape.
	# 	"""
	# 	ret = (pic1.data.shape == pic2.data.shape)*1

	# 	return Block.new_integer(ret)
