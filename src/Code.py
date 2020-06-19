# Jazz (c) 2018-2020 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses

from Block import Block


class Code(Block):
	"""
	This is class defines a code snippet using any Function descendants. A code is a block and is just a tuple of blocks
	(mostly functions). Code objects are created by Search or Field.compile() and run by a Core (possibly inside a Field).
	"""

	def __init__(self, tup):
		self.data = tup
		self.type = Block.type_no_error
