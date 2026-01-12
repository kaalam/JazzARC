# Jazz (c) 2018-2026 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses

from Block	   import Block
from Container import Container
from Field	   import Field
from Source	   import Source


class CodeBase(Container):
	"""
	This contains three Container objects of the same length for Code, Source and name items.

	Names must be unique strings at least three characters long.

	Codes and sources can be returned by name.

	The whole collection can be loaded and saved from/to text files.

	The object can be initialized from a text file on construction. In case a file_name is given,
	abstract_relation cannot be None since compilation is necessary.
	"""

	def __init__(self, file_name=None, abstract_relation=None):
		self.code	= Container()
		self.source	= Container()
		self.name	= Container()
		self.sample	= Container()

		self.idx_item = {}

		if file_name is not None:
			self.load(file_name, abstract_relation)


	def __getitem__(self, i):
		"""
		This overrides Container.__getitem__()
		"""
		return (self.code[i], self.source[i], self.name[i], self.sample[i])


	def num_items(self):
		"""
		This overrides Container.num_items()
		"""
		return len(self.code.data)


	def add(self, code, source, name, sample):
		"""
		This overrides Container.add()
		"""
		if not isinstance(name, str) or len(name) < 3 or name in self.idx_item:
			raise NameError

		self.idx_item[name] = self.num_items()

		self.code.data.append(code)
		self.source.data.append(source)
		self.name.data.append(name)
		self.sample.data.append(sample)


	def code_by_name(self, name):
		"""
		Return the code for an existing name.
		"""
		return self.code[self.idx_item[name]]


	def source_by_name(self, name):
		"""
		Return the source for an existing name.
		"""
		return self.source[self.idx_item[name]]


	def sample_by_name(self, name):
		"""
		Return the sample for an existing name.
		"""
		return self.sample[self.idx_item[name]]


	def load(self, file_name, abstract_relation):
		"""
		This loads and compiles a collection of code snippets from a text file.

		The snippets are appended to the content already in the CodeBase. Names cannot
		match any of the names already in the CodeBase.

		It requires a Relation Class for a language definition for compiling.
		"""
		field = Field(abstract_relation)

		with open(file_name, 'r') as f:
			txt = f.read().splitlines()

		if not txt[0].startswith('.bopDB'):
			raise ValueError

		name = None
		for line in txt[3:]:
			if name is None:
				if line == '.eof.':
					return

				name   = line
				source = None
				sample = None
			elif line.startswith('-'):
				source_tuple = ()
			elif line == '':
				if source is None:
					source = Source(source_tuple)
					code   = field.compile(source)
					if code.type == Block.type_error:
						raise ValueError
				else:
					self.add(code, source, name, sample)
					name = None
			else:
				if source is None:
					source_tuple = source_tuple + (line,)
				else:
					try:
						pic = eval(line)
					except Exception:
						raise ValueError

					sample = Block.new_picture(list_of_list=pic)

		raise ValueError


	def save(self, file_name):
		"""
		This saves the source code of all Code objects in this Container to a text file.

		The file is overridden if it exists.
		"""
		with open(file_name, 'w') as f:
			f.write('.bopDB: ' + file_name + '\n\n\n')

			names = sorted(self.name.data)
			for name in names:
				source = self.source_by_name(name)
				sample = self.sample_by_name(name)

				f.write(name + '\n')
				f.write('-' * len(name) + '\n')
				for statement in source.data:
					f.write(statement + '\n')

				f.write('\n')
				f.write(str(sample.data.tolist()) + '\n')
				f.write('\n')

			f.write('.eof.')
