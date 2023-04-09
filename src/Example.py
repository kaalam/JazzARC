# Jazz (c) 2018-2023 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses

from Answer	  import Answer
from Bond	  import Bond
from Question import Question


class Example(Bond):
	"""
	Inherits Bond to make it link Questions to Answers (with an extra is_test argument).

	This class contains one Question and its Answer.

	The class containing more than one (Question, Answer) pairs sharing the same Code is Problem.
	"""

	def __init__(self, q_list_of_list, a_list_of_list, is_test):
		if is_test and a_list_of_list is None:
			self.data = (Question(q_list_of_list), None, True)
		else:
			self.data = (Question(q_list_of_list), Answer(a_list_of_list), is_test)


	def from_kind():
		"""
		This defines the origin Kind of the Relation.
		"""
		return Question


	def to_kind():
		"""
		This defines the destination Kind of the Relation.
		"""
		return Answer


	def function_classes():
		"""
		This merges the unique classes in the Kinds of this Relation and returns them as a set.
		"""
		return set(Example.from_kind().function_classes() + Example.to_kind().function_classes())
