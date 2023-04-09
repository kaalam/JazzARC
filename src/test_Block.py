# Jazz (c) 2018-2023 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses

import numpy as np

from Block import Block


def test_instancing():
	c = Block.new_integer(7)

	assert isinstance(c, Block)
	assert c.type == Block.type_integer
	assert c.data == 7
	assert isinstance(c.data, int)

	p = Block.new_int_pair((3, 6))

	assert isinstance(p, Block)
	assert p.type == Block.type_int_pair
	assert p.data == (3, 6)
	assert isinstance(p.data, tuple)
	assert isinstance(p.data[0], int)

	e = Block.new_error('Stack overflow')

	assert isinstance(e, Block)
	assert e.type == Block.type_error
	assert e.data == 'Stack overflow'

	f = Block.new_function(len)

	assert isinstance(f, Block)
	assert f.type == Block.type_function
	assert callable(f.data)

	m = Block.new_NESW((0, 1, 1, 0))

	assert isinstance(m, Block)
	assert m.type == Block.type_NESW
	assert m.data == (0, 1, 1, 0)
	assert isinstance(m.data, tuple)
	assert isinstance(m.data[0], int)

	i = np.array([[1, 2, 3]], dtype=np.int32)
	k = i > 2

	t = Block.new_pattern(i, k)
	assert isinstance(t, Block)
	assert t.type == Block.type_pattern
	assert isinstance(t.data, tuple)

	q = Block.new_picture(list_of_list=[[30, 60, 90], [8, 0, 1]])

	assert isinstance(q, Block)
	assert q.type == Block.type_picture
	assert isinstance(q.data, np.ndarray)

	u = np.array([[1, 0], [0, -2]], dtype=np.int32)

	d = Block.new_picture(np_arr=u)

	assert isinstance(d, Block)
	assert d.type == Block.type_picture
	assert isinstance(d.data, np.ndarray)

	n = Block.new_pictures((i, u))

	assert isinstance(n, Block)
	assert n.type == Block.type_pictures
	assert isinstance(n.data, tuple)
	assert isinstance(n.data[0], np.ndarray)

	v = Block.new_vector([3, 2, 1])

	assert isinstance(v, Block)
	assert v.type == Block.type_vector
	assert v.data == [3, 2, 1]
	assert isinstance(v.data, list)
	assert isinstance(v.data[0], int)


def test_calling_types():
	assert Block.type_integer()	 == 'integer'
	assert Block.type_int_pair() == 'int_pair'
	assert Block.type_core()	 == 'core'
	assert Block.type_error()	 == 'error'
	assert Block.type_function() == 'function'
	assert Block.type_NESW()	 == 'NESW'
	assert Block.type_no_error() == 'no_error'
	assert Block.type_nothing()	 == 'nothing'
	assert Block.type_pattern()	 == 'pattern'
	assert Block.type_picture()	 == 'picture'
	assert Block.type_pictures() == 'pictures'
	assert Block.type_vector()	 == 'vector'
