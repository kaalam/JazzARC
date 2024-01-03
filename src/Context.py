# Jazz (c) 2018-2024 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses

import json
import os

from Container import Container
from Example   import Example
from Problem   import Problem


class Context(Container):
	"""
	This is the class storing the dataset of all problems.

	Problems are loaded on construction from a folder containing only ARC-compatible json files.

	The path to that folder is given as an argument.
	"""

	def __init__(self, path=None):
		Container.__init__(self)
		if path is not None:
			self.load_data(path)


	def load_data(self, path):
		"""
		This is the loading function. It is called by the constructor.
		"""
		self.file_names = os.listdir(path)
		self.file_names.sort()

		for fn in self.file_names:
			with open(path + '/' + fn) as f:
				item = json.load(f)

			p = Problem()
			for case in item['train']:
				p.add(Example(case['input'], case['output'], False))

			assert p.num_items() >= 2 and p.num_items() <= 10
			assert len(item['test']) >= 1 and len(item['test']) <= 3

			for case in item['test']:
				p.add(Example(case['input'], case['output'], True))

			self.add(p)


	def get_problem_startswith(self, starts_with):
		"""
		Returns a problem by incomplete match (using startswith) of its file name.

		If more than one file names match, the first found is returned.
		"""
		for i, fn in enumerate(self.file_names):
			if fn.startswith(starts_with):
				return self.data[i]
