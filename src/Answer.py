# Jazz (c) 2018-2026 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses

import numpy as np

from Block	 import Block
from BopBack import BopBack


class Answer(Block):
	"""
	This is the Kind containing answers (Block.type_picture instances).

	Since this PoC has no Kind implementation, an Answer is a Block, but it has an additional function_classes() method
	defining DSL functions this Kind requires to be in a Field that computes relations to of from this Kind.
	"""

	def __init__(self, list_of_list):
		self.data = np.array(list_of_list, dtype=np.int32)
		self.type = Block.type_picture


	def function_classes():
		"""
		This defines DSL function collections that should be included in a Field that computes Answers.

		Note function_classes() has no self argument. It is a method of the abstract Kind while Answer() is just a way
		to create data storage.

		The classes passed in this list are also abstract classes that contain @bop_function decorated methods.
		"""
		return [BopBack]
