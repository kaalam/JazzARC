# Jazz (c) 2018-2023 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses

from Container import Container


class Problem(Container):
	"""
	This is class is a Container of Example.

	A Problem should run all its Example objects using the same Code.

	Problems are not loaded or saved individually, they are stored in a Context.
	"""

	def __init__(self):
		Container.__init__(self)
