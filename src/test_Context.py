# Jazz (c) 2018-2025 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses

from Answer	  import Answer
from Context  import Context
from Example  import Example
from Problem  import Problem
from Question import Question


def test_Context():
	ctx = Context('data')

	for prob in ctx:
		assert isinstance(prob, Problem)
		for example in prob:
			assert isinstance(example, Example)
			q, a, is_test = example.data
			assert isinstance(q, Question)
			assert isinstance(a, Answer)
			assert isinstance(is_test, bool)
