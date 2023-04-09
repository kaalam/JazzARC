# Jazz (c) 2018-2023 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses

from Block import Block


class Source(Block):
	"""
	A source is a Block of type tuple (of strings) that can be compiled into a Code object by a Field.
	"""

	def __init__(self, tuple_of_string):
		self.data = tuple_of_string
		self.type = Block.type_no_error
