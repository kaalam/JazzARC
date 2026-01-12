# Jazz (c) 2018-2026 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses

from Block import Block


class Bond(Block):
	"""
	Defines the only Relation in this PoC between two Kinds: from_kind and to_kind.

	It is only a Block descendant because there is no class Relation in this PoC, just like
	Questions and Answers inherit Block instead of Kind. Unlike in the latter, there is no
	point of storing anything in a Block and it uses normal Python properties.
	"""

	def __init__(self, from_kind, to_kind):
		self.from_kind = from_kind
		self.to_kind   = to_kind
		self.type	   = Block.type_no_error


	def function_classes():
		"""
		This merges the unique classes in the Kinds of this Relation.

		This is the parent of the actual Bonds (Example) since this class is generic, it cannot be implemented
		and must be implemented in the descendant.
		"""
		raise NotImplementedError
