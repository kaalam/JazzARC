# Jazz (c) 2018-2026 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses


class Container:
	"""
	This is the common ancestor of most classes. Unlike in real Jazz, python includes enough data structures out of the box,
	for the moment, a Container is just a list of items of the same type (mandatory, but not checked).
	"""

	def __init__(self):
		self.data = []


	def __getitem__(self, i):
		"""
		This magic python method not just supports the [] indexing but, more importantly, makes Container an iterator.
		E.g., `for prob in context:` works thanks to this.
		"""
		return self.data[i]


	def num_items(self):
		"""
		Returns len() of the list.
		"""
		return len(self.data)


	def add(self, item):
		"""
		This is the way to add something to a container, for the moment __setitem__() and __delitem__() are not used.
		It also does not check types. Nevertheless, `assert type(item) == type(self.data[0])` should be assumed.
		"""
		self.data.append(item)
