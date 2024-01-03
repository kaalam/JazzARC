# Jazz (c) 2018-2024 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses

import numpy as np

from Block	  import Block
from Function import bop_function


class BopForward():
	"""
	This abstract class is the storage of @bop_function decorated Bebop functions for all the DSL code.

	Method names are searched by Bebop and are lowercase class name underscore opcode: bopforward_opcode

	All opcodes follow the same convention:

	They start with all the argument types underscore separated as 3-4 letter abbreviations:

	type_integer		int
	type_int_pair		intp
	type_NESW			nesw
	type_pattern		pat
	type_picture		pic
	type_pictures		pics
	type_vector			vec

	When types repeat, they are prefixed by # times: pic_pic_ is 2pic_, etc.

	Then, a short lowercase underscore separated description.

	Finally, the return type is omitted if it is the same as the first argument type, otherwise the name finishes with _as_type, where
	type is one of the abbreviations above.

	E.g.,
		pic_rotate_90_ccw			: One argument, same return type
		pic_fork_on_v_axis_as_pics	: One argument, different return type
		pic_intp_swap_colors		: Two different arguments, return as the first one.
		2pat_merge					: Two identical arguments, return as the first one.
		2pat_merge_as_pic			: Two identical arguments, return as a different type.

	"""
	@bop_function(arg_types=[Block.type_pattern, Block.type_pattern], ret_type=Block.type_pattern)
	def bopforward_2pat_merge(pat1, pat2):
		"""
		Merges two patterns and returns the result as a pattern.

		In case of overlap, it takes the maximum value.
		"""
		pic1, mask1 = pat1.data
		pic2, mask2 = pat2.data

		if pic1.shape != pic2.shape:
			return Block.new_error('pic1.shape != pic2.shape')

		pic	 = np.maximum(pic1*mask1, pic2*mask2)
		mask = np.logical_or(mask1, mask2)

		return Block.new_pattern(pic, mask)


	@bop_function(arg_types=[Block.type_pattern, Block.type_pattern], ret_type=Block.type_picture)
	def bopforward_2pat_merge_as_pic(pat1, pat2):
		"""
		Merges two patterns and returns the result as a pic.

		In case of overlap, it takes the maximum value.
		"""
		pic1, mask1 = pat1.data
		pic2, mask2 = pat2.data

		if pic1.shape != pic2.shape:
			return Block.new_error('pic1.shape != pic2.shape')

		return Block.new_picture(np_arr=np.maximum(pic1*mask1, pic2*mask2))


	@bop_function(arg_types=[Block.type_pattern, Block.type_pattern], ret_type=Block.type_picture)
	def bopforward_2pat_merge_if_disjoint_as_pic(pat1, pat2):
		"""
		Merges two patterns only if not overlapping (otherwise returns the first one) as a pic.
		"""
		pic1, mask1 = pat1.data
		pic2, mask2 = pat2.data

		if pic1.shape != pic2.shape:
			return Block.new_error('pic1.shape != pic2.shape')

		if np.any(np.logical_and(mask1, mask2)):
			return Block.new_picture(np_arr=pic1)

		return Block.new_picture(np_arr=pic1*mask1 + pic2*mask2)


	@bop_function(arg_types=[Block.type_picture, Block.type_picture], ret_type=Block.type_picture)
	def bopforward_2pic_and_masks_to_1(pic1, pic2):
		"""
		Gets the and-operation from masks (color != 0) by cells from two pictures return as 0 or 1.
		"""
		if pic1.data.shape != pic2.data.shape:
			return Block.new_error('pic1.shape != pic2.shape')

		return Block.new_picture(np_arr=np.array(np.logical_and(pic1.data != 0, pic2.data != 0), dtype=np.int32))


	@bop_function(arg_types=[Block.type_picture, Block.type_picture], ret_type=Block.type_picture)
	def bopforward_2pic_cbind(pic1, pic2):
		"""
		Builds the picture [pic1|pic2] like an R cbind instruction.
		"""
		if pic1.data.shape[0] != pic2.data.shape[0]:
			return Block.new_error('pic1.shape and pic2.shape not same height')

		return Block.new_picture(np_arr=np.hstack([pic1.data, pic2.data]))


	@bop_function(arg_types=[Block.type_picture, Block.type_picture], ret_type=Block.type_picture)
	def bopforward_2pic_maximum(pic1, pic2):
		"""
		Gets the maximum color by cells from two pictures with the same shape.
		"""
		if pic1.data.shape != pic2.data.shape:
			return Block.new_error('pic1.shape != pic2.shape')

		return Block.new_picture(np_arr=np.maximum(pic1.data, pic2.data))


	@bop_function(arg_types=[Block.type_picture, Block.type_picture], ret_type=Block.type_picture)
	def bopforward_2pic_multiply(brick, layout):
		"""
		Returns an image repeating then brick over each non-zero pixel of the layout.
		"""
		mask = np.repeat(np.repeat(layout.data == 0, brick.data.shape[0], axis=0), brick.data.shape[1], axis=1)
		pic	 = np.tile(brick.data, layout.data.shape)

		pic[mask] = 0

		return Block.new_picture(np_arr=pic)


	@bop_function(arg_types=[Block.type_picture, Block.type_picture], ret_type=Block.type_picture)
	def bopforward_2pic_rbind(pic1, pic2):
		"""
		Builds the picture [pic1] on top of [pic2] like an R rbind instruction.
		"""
		if pic1.data.shape[1] != pic2.data.shape[1]:
			return Block.new_error('pic1.shape and pic2.shape not same width')

		return Block.new_picture(np_arr=np.vstack([pic1.data, pic2.data]))


	@bop_function(arg_types=[Block.type_picture, Block.type_picture], ret_type=Block.type_picture)
	def bopforward_2pic_recolor_any_rtl(pic, rec_pat):
		"""
		Recolors (== anything non-black recolors as what is non-black in rec_pat) pic tiling rec_pat right-to-left.
		"""
		dy, dx = pic.data.shape
		oy, ox = rec_pat.data.shape

		nx = int((dx + ox - 1)/ox)
		ny = int((dy + oy - 1)/oy)

		new_pic = np.tile(rec_pat.data, (ny, nx))

		if new_pic.shape[0] > dy:
			new_pic = new_pic[range(dy), :]

		if new_pic.shape[1] > dx:
			ox = new_pic.shape[1] - dx
			new_pic = new_pic[:, range(ox, dx + ox)]			# This the right-to-left! Adjust the right margin, not the left one

		mask1 = (pic.data > 0)*(new_pic > 0)
		mask2 = np.logical_not(mask1)

		new_pic = new_pic*mask1 + pic.data*mask2

		return Block.new_picture(np_arr=new_pic)


	@bop_function(arg_types=[Block.type_picture, Block.type_picture], ret_type=Block.type_picture)
	def bopforward_2pic_tile_all(canvas, rep_pat):
		"""
		Fills a canvas (overriding any state other than shape) by tiling with a pattern both horizontally and vertically.
		"""
		dy, dx = canvas.data.shape
		oy, ox = rep_pat.data.shape

		nx = int((dx + ox - 1)/ox)
		ny = int((dy + oy - 1)/oy)

		new_pic = np.tile(rep_pat.data, (ny, nx))

		return Block.new_picture(np_arr=new_pic[range(dy), :][:, range(dx)])


	@bop_function(arg_types=[Block.type_picture, Block.type_picture], ret_type=Block.type_picture)
	def bopforward_2pic_xor_masks_to_1(pic1, pic2):
		"""
		Gets the xor-operation from masks (color != 0) by cells from two pictures return as 0 or 1.
		"""
		if pic1.data.shape != pic2.data.shape:
			return Block.new_error('pic1.shape != pic2.shape')

		return Block.new_picture(np_arr=np.array(np.logical_xor(pic1.data != 0, pic2.data != 0), dtype=np.int32))


	@bop_function(arg_types=[Block.type_integer], ret_type=Block.type_picture)
	def bopforward_int_black_box_as_pic(ll):
		"""
		Returns a block box of size ll*ll.
		"""
		pic = np.zeros((ll.data, ll.data), dtype=np.int32)

		return Block.new_picture(np_arr=pic)


	@bop_function(arg_types=[Block.type_pattern], ret_type=Block.type_picture)
	def bopforward_pat_as_pic(pat):
		"""
		Returns pattern as picture.
		"""
		pic, mask = pat.data

		return Block.new_picture(np_arr=pic*mask)


	@bop_function(arg_types=[Block.type_pattern], ret_type=Block.type_pattern)
	def bopforward_pat_flip_left_right(pat):
		"""
		Flips a pattern in the up/down direction.
		"""
		pic, mask = pat.data

		pic	 = np.fliplr(pic)
		mask = np.fliplr(mask)

		return Block.new_pattern(pic, mask)


	@bop_function(arg_types=[Block.type_pattern], ret_type=Block.type_pattern)
	def bopforward_pat_flip_up_down(pat):
		"""
		Flips a pattern in the up/down direction.
		"""
		pic, mask = pat.data

		pic	 = np.flipud(pic)
		mask = np.flipud(mask)

		return Block.new_pattern(pic, mask)


	@bop_function(arg_types=[Block.type_pattern, Block.type_NESW], ret_type=Block.type_pattern)
	def bopforward_pat_nesw_drag_all(pat, nesw):
		"""
		Drags a pattern by a steps defined as (N, E, S, W).

		Dragging means moving (by adding unselected black on one border) and just forgetting what gets lost over the horizon.
		"""
		N, E, S, W = nesw.data

		pic, mask = pat.data

		hh, ww = pic.shape

		if N > 0:
			pic	 = np.vstack([pic [range(N, hh), :], np.zeros((N, ww), dtype=np.int32)])
			mask = np.vstack([mask[range(N, hh), :], np.zeros((N, ww), dtype=np.bool)])

		if E > 0:
			pic	 = np.hstack([np.zeros((hh, E), dtype=np.int32), pic[:, range(ww - E)]])
			mask = np.hstack([np.zeros((hh, E), dtype=np.bool), mask[:, range(ww - E)]])

		if S > 0:
			pic	 = np.vstack([np.zeros((S, ww), dtype=np.int32), pic[range(hh - S), :]])
			mask = np.vstack([np.zeros((S, ww), dtype=np.bool), mask[range(hh - S), :]])

		if W > 0:
			pic	 = np.hstack([pic [:, range(W, ww)], np.zeros((hh, W), dtype=np.int32)])
			mask = np.hstack([mask[:, range(W, ww)], np.zeros((hh, W), dtype=np.bool)])

		return Block.new_pattern(pic, mask)


	@bop_function(arg_types=[Block.type_picture, Block.type_int_pair, Block.type_int_pair], ret_type=Block.type_picture)
	def bopforward_pic_2intp_crop(pic, ori, siz):
		"""
		Crops a pic by ori.
		"""
		oy, ox = ori.data
		dy, dx = siz.data

		if pic.data.shape[0] < oy + dy or pic.data.shape[1] < ox + dx:
			return Block.new_error('Image too small for required crop')

		return Block.new_picture(np_arr=pic.data[:, range(ox, ox + dx)][range(oy, oy + dy), :])


	@bop_function(arg_types=[Block.type_picture], ret_type=Block.type_pattern)
	def bopforward_pic_all_as_pat(pic):
		"""
		Takes a pic and returns a pattern with all nonzero cells selected.
		"""
		mask = np.logical_and(pic.data, 1)

		return Block.new_pattern(pic.data, mask)


	@bop_function(arg_types=[Block.type_picture], ret_type=Block.type_vector)
	def bopforward_pic_all_colors_as_vec(pic):
		"""
		Creates a vector of all non-zero colors.
		"""
		return Block.new_vector([cc for cc in np.unique(pic.data) if cc > 0])


	@bop_function(arg_types=[Block.type_picture], ret_type=Block.type_pictures)
	def bopforward_pic_autohalves_as_pics(pic):
		"""
		Splits a pic on largest axis either left and right or top and bottom.
		"""
		hh, ww = pic.data.shape

		if hh > ww:
			vv = int(hh/2)

			top = pic.data[range(vv), :]
			bot = pic.data[range(hh - vv, hh), :]

			return Block.new_pictures((top, bot))

		vv = int(ww/2)

		left  = pic.data[:, range(vv)]
		right = pic.data[:, range(ww - vv, ww)]

		return Block.new_pictures((left, right))


	@bop_function(arg_types=[Block.type_picture], ret_type=Block.type_integer)
	def bopforward_pic_base_height_as_int(pic):
		"""
		Returns the y of the last completely black row starting in reverse and stopping when color is found.
		"""
		hh = pic.data.shape[0]
		for yy in reversed(range(hh)):
			if np.any(pic.data[yy, :] != 0):
				return Block.new_integer(yy + 1)
		return Block.new_integer(0)


	@bop_function(arg_types=[Block.type_picture], ret_type=Block.type_picture)
	def bopforward_pic_corners(pic):
		"""
		Returns a black picture with the same shape as pic and the corners as 1.
		"""
		pic = np.zeros(pic.data.shape, dtype=np.int32)
		ey, ex = pic.shape

		pic[0,		0]		= 1
		pic[0, 		ex - 1] = 1
		pic[ey - 1, 0]		= 1
		pic[ey - 1, ex - 1] = 1

		return Block.new_picture(np_arr=pic)


	@bop_function(arg_types=[Block.type_picture], ret_type=Block.type_picture)
	def bopforward_pic_distinct_border_colors(pic):
		"""
		Returns a vertical or horizontal vector (whatever is longer) with the distinct border colors.
		"""
		ey, ex = pic.data.shape

		hh = [pic.data[0, 0]]
		for x in range(1, ex):
			if pic.data[0, x] != hh[-1]:
				hh.append(pic.data[0, x])

		vv = [pic.data[0, 0]]
		for y in range(1, ey):
			if pic.data[y, 0] != vv[-1]:
				vv.append(pic.data[y, 0])

		if len(hh) > len(vv):
			pic = np.array([hh])
		else:
			pic = np.array([vv]).transpose()

		return Block.new_picture(np_arr=pic)


	@bop_function(arg_types=[Block.type_picture], ret_type=Block.type_picture)
	def bopforward_pic_filter_axes(pic):
		"""
		Returns an image with anything other than vertical or horizontal axes as black.
		"""
		pic = pic.data.copy()

		hh, ww = pic.shape

		mask = np.zeros((hh, ww), dtype=np.int32)

		for xx in range(ww):
			cc = pic[0, xx]
			if np.all(pic[:, xx] == cc):
				mask[:, xx] = 1

		for yy in range(hh):
			cc = pic[yy, 0]
			if np.all(pic[yy, :] == cc):
				mask[yy, :] = 1

		pic[mask == 0] = 0

		return Block.new_picture(np_arr=pic)


	@bop_function(arg_types=[Block.type_picture], ret_type=Block.type_picture)
	def bopforward_pic_filter_leastfreq_col(pic):
		"""
		Returns an image with anything other than the least frequent color as black.
		"""
		pic = pic.data.copy()

		cc = np.unique(pic, return_counts=True)
		fc = cc[0][np.argmin(cc[1])]

		pic[pic != fc] = 0

		return Block.new_picture(np_arr=pic)


	@bop_function(arg_types=[Block.type_picture], ret_type=Block.type_picture)
	def bopforward_pic_filter_mostfreq_col(pic):
		"""
		Returns an image with anything other than the most frequent color as black.
		"""
		pic = pic.data.copy()

		cc = np.unique(pic, return_counts=True)
		fc = cc[0][np.argmax(cc[1])]

		pic[pic != fc] = 0

		return Block.new_picture(np_arr=pic)


	@bop_function(arg_types=[Block.type_picture], ret_type=Block.type_pictures)
	def bopforward_pic_fork_by_color_as_pics(pic):
		"""
		Splits a pic into as many colors as found, cropped to the boundary of each color pattern.
		"""
		pics = ()
		for color in range(1, 10):
			mask	= pic.data == color
			new_pic = pic.data[np.ix_(mask.any(1), mask.any(0))]

			if new_pic.shape[0] > 0:
				pics = pics + (new_pic,)

		return Block.new_pictures(pics)


	@bop_function(arg_types=[Block.type_picture], ret_type=Block.type_pictures)
	def bopforward_pic_fork_color_rest_black_as_pics(pic):
		"""
		Works like pic_fork_by_color_as_pics() but keeping the rest as a black background.
		"""
		pics = ()
		for color in range(1, 10):
			mask = pic.data == color

			if np.sum(mask) > 0:
				pics = pics + (pic.data.copy()*mask,)

		return Block.new_pictures(pics)


	@bop_function(arg_types=[Block.type_picture], ret_type=Block.type_pictures)
	def bopforward_pic_fork_on_auto_grid_as_pics(pic):
		"""
		Locates a grid of any color and pushes each cell in the grid as a pic in a Block.type_pictures.
		"""
		hh, ww = pic.data.shape

		def auto_locate_grid_color():
			for xx in range(1, ww - 1):
				cc = int(pic.data[0, xx])
				if np.all(pic.data[:, xx] == cc):
					return cc
			for yy in range(1, hh - 1):
				cc = int(pic.data[yy, 0])
				if np.all(pic.data[yy, :] == cc):
					return cc
			return 0

		cc = auto_locate_grid_color()

		def next_black_row(oy):
			while oy < hh:
				if np.all(pic.data[oy, :] == cc):
					return oy
				oy += 1
			return oy

		def next_black_col(ox):
			while ox < ww:
				if np.all(pic.data[:, ox] == cc):
					return ox
				ox += 1
			return ox

		ret = ()

		oy = 0
		while oy < hh:
			ey = next_black_row(oy + 1)
			if ey - oy > 0:
				ox = 0
				while ox < ww:
					ex = next_black_col(ox + 1)
					if ex - ox > 0:
						ret = ret + (pic.data[range(oy, ey), :][:, range(ox, ex)],)
					ox = ex + 1
			oy = ey + 1

		return Block.new_pictures(ret)


	@bop_function(arg_types=[Block.type_picture], ret_type=Block.type_pictures)
	def bopforward_pic_fork_on_h_axis_as_pics(pic):
		"""
		Splits a pic into top and bottom and returns both as Block.type_pictures.
		"""
		hh = pic.data.shape[0]
		vv = int(hh/2)

		top = pic.data[range(vv), :]
		bot = pic.data[range(hh - vv, hh), :]

		return Block.new_pictures((top, bot))


	@bop_function(arg_types=[Block.type_picture], ret_type=Block.type_pictures)
	def bopforward_pic_fork_on_v_axis_as_pics(pic):
		"""
		Splits a pic into left and right and returns both as Block.type_pictures.
		"""
		ww = pic.data.shape[1]
		vv = int(ww/2)

		left  = pic.data[:, range(vv)]
		right = pic.data[:, range(ww - vv, ww)]

		return Block.new_pictures((left, right))


	@bop_function(arg_types=[Block.type_picture, Block.type_integer], ret_type=Block.type_picture)
	def bopforward_pic_int_copy_border(pic, tt):
		"""
		Copies the border of the image in all directions tt times.
		"""
		pic = pic.data.copy()

		for _ in range(tt.data):
			ww  = pic.shape[1]
			pic = np.hstack([pic[:, [0]], pic, pic[:, [ww - 1]]])
			hh  = pic.shape[0]
			pic = np.vstack([pic[[0], :], pic, pic[[hh - 1], :]])

		return Block.new_picture(np_arr=pic)


	@bop_function(arg_types=[Block.type_picture, Block.type_integer], ret_type=Block.type_picture)
	def bopforward_pic_int_empty_border(pic, cc):
		"""
		Returns a black image with the same shape and a border of color cc.
		"""
		hh, ww = pic.data.shape

		pic = np.zeros((hh, ww), dtype=np.int32)

		pic[:, 0]	   = cc.data
		pic[:, ww - 1] = cc.data
		pic[0, :]	   = cc.data
		pic[hh - 1, :] = cc.data

		return Block.new_picture(np_arr=pic)


	@bop_function(arg_types=[Block.type_picture, Block.type_integer], ret_type=Block.type_picture)
	def bopforward_pic_int_filter_color(pic, cc):
		"""
		Returns an image with anything other than cc as black.
		"""
		pic = pic.data.copy()
		pic[pic != cc.data] = 0

		return Block.new_picture(np_arr=pic)


	@bop_function(arg_types=[Block.type_picture, Block.type_integer], ret_type=Block.type_picture)
	def bopforward_pic_int_recolor_all(pic, col):
		"""
		Takes a pic and a color and returns a new pic with anything non-black as the color.
		"""
		pic = pic.data.copy()

		pic[pic != 0] = col.data

		return Block.new_picture(np_arr=pic)


	@bop_function(arg_types=[Block.type_picture, Block.type_integer], ret_type=Block.type_picture)
	def bopforward_pic_int_slide_rows_west(pic, hh):
		"""
		Slides each row to the left by one extra unit, leaving the base as is. The base is the row above hh.
		"""
		pic = pic.data.copy()
		ww  = pic.shape[1]

		for yy, xx in zip(reversed(range(hh.data - 1)), range(1, hh.data)):
			if xx < ww:
				pic[yy, :] = np.hstack([pic[yy, xx:ww], np.zeros((xx), dtype=np.int32), ])
			else:
				pic[yy, :] = np.zeros(ww, dtype=np.int32)

		return Block.new_picture(np_arr=pic)


	@bop_function(arg_types=[Block.type_picture, Block.type_integer], ret_type=Block.type_picture)
	def bopforward_pic_int_zoom_in(pic, tt):
		"""
		Zooms into the same picture (repeating each pixel tt*tt).
		"""
		dy, dx = pic.data.shape
		tt	   = tt.data

		pic = np.repeat(np.repeat(pic.data, tt, axis=0), tt, axis=1)

		return Block.new_picture(np_arr=pic)


	@bop_function(arg_types=[Block.type_picture, Block.type_integer], ret_type=Block.type_picture)
	def bopforward_pic_int_zoom_out(pic, tt):
		"""
		Zooms out of the same picture by taking maximum of each tt*tt square.
		"""
		dy, dx = pic.data.shape
		tt	   = tt.data

		ex = int(dx/tt)
		ey = int(dy/tt)

		if (ex*tt != dx or ey*tt != dy):
			return Block.new_error('Wrong shapes for scale')

		aa = dx*dy
		h1 = int(aa/tt)
		h2 = int(h1/ex)
		h3 = int(h1/tt)

		pic = pic.data.reshape(h1, tt)
		pic = np.amax(pic, 1).reshape(h2, ex).transpose().reshape(h3, tt)
		pic = np.amax(pic, 1).reshape(ex, ey).transpose()

		return Block.new_picture(np_arr=pic)


	@bop_function(arg_types=[Block.type_picture, Block.type_int_pair], ret_type=Block.type_pictures)
	def bopforward_pic_intp_fork_on_shape_as_pics(pic, output_shape):
		"""
		Splits a pic vertically or horizontally to match the output_shape and returns all as Block.type_pictures.
		"""
		oh, ow = output_shape.data
		hh, ww = pic.data.shape

		ret = ()

		if oh == hh:
			tt = int(ww/ow)
			if ww != ow*tt:
				return Block.new_error('Pic width not a multiple of output')

			for ii in range(tt):
				ox  = ii*ow
				ret = ret + (pic.data[:, range(ox, ox + ow)],)

		elif ow == ww:
			tt = int(hh/oh)
			if hh != oh*tt:
				return Block.new_error('Pic height not a multiple of output')

			for ii in range(tt):
				oy  = ii*oh
				ret = ret + (pic.data[range(oy, oy + oh), :],)

		else:
			return Block.new_error('Neither height nor width match output')

		return Block.new_pictures(ret)


	@bop_function(arg_types=[Block.type_picture, Block.type_int_pair], ret_type=Block.type_picture)
	def bopforward_pic_intp_recolor(pic, col_pair):
		"""
		Takes a pic and a tuple of two colors and returns a new pic with the first color recolored as the second.
		"""
		c1, c2 = col_pair.data
		pic	   = pic.data.copy()

		pic[pic == c1] = c2

		return Block.new_picture(np_arr=pic)


	@bop_function(arg_types=[Block.type_picture, Block.type_int_pair], ret_type=Block.type_picture)
	def bopforward_pic_intp_select_columns(pic, cc):
		"""
		Takes a pic and a column range and returns just the columns as a new pic.
		"""
		ox, ex = cc.data

		return Block.new_picture(np_arr=pic.data[:, range(ox, ex)])


	@bop_function(arg_types=[Block.type_picture, Block.type_int_pair], ret_type=Block.type_picture)
	def bopforward_pic_intp_swap_colors(pic, col_pair):
		"""
		Takes a pic and a tuple of two colors and returns a new pic with the colors swapped.
		"""
		c1, c2 = col_pair.data

		pic = pic.data.copy()

		pic[pic == c1] = 99
		pic[pic == c2] = c1
		pic[pic == 99] = c2

		return Block.new_picture(np_arr=pic)


	@bop_function(arg_types=[Block.type_picture, Block.type_int_pair], ret_type=Block.type_picture)
	def bopforward_pic_intp_zoom_fit(pic, out_shape):
		"""
		Takes a pic and an output shape and does either zoom in or out to the shape if possible.
		"""
		oy, ox = pic.data.shape
		ey, ex = out_shape.data

		if oy < ey:
			tt = int(ey/oy)

			if ey != oy*tt or ex != ox*tt:
				return Block.new_error('Not an integer scale for zoom_in')

			return BopForward.bopforward_pic_int_zoom_in.data(pic, Block.new_integer(tt))

		if oy > ey:
			tt = int(oy/ey)

			if oy != ey*tt or ox != ex*tt:
				return Block.new_error('Not an integer scale for zoom_out')

			return BopForward.bopforward_pic_int_zoom_out.data(pic, Block.new_integer(tt))

		if ox != ex:
			return Block.new_error('X scale != Y scale')

		return pic


	@bop_function(arg_types=[Block.type_picture, Block.type_NESW], ret_type=Block.type_picture)
	def bopforward_pic_nesw_extend(pic, nesw):
		"""
		Extends a pic adding zero by (N, E, S, W).
		"""
		N, E, S, W = nesw.data

		pic = pic.data.copy()

		ww = pic.shape[1]

		if N > 0:
			pic = np.vstack([np.zeros((N, ww), dtype=np.int32), pic])

		if S > 0:
			pic = np.vstack([pic, np.zeros((S, ww), dtype=np.int32)])

		hh = pic.shape[0]

		if W > 0:
			pic = np.hstack([np.zeros((hh, W), dtype=np.int32), pic])

		if E > 0:
			pic = np.hstack([pic, np.zeros((hh, E), dtype=np.int32)])

		return Block.new_picture(np_arr=pic)


	@bop_function(arg_types=[Block.type_picture, Block.type_NESW], ret_type=Block.type_picture)
	def bopforward_pic_nesw_gravity(pic, nesw):
		"""
		Moves all pixels in the nesw direction only if the next pixel is black..
		"""

		N, E, S, W = nesw.data

		pic = pic.data.copy()

		ww = pic.shape[1]
		hh = pic.shape[0]

		if N > 0:
			for y in range(hh - N):
				for x in range(ww):
					if pic[y, x] == 0:
						pic[y, x] = pic[y + N, x]
						pic[y + N, x] = 0

		if E > 0:
			for x in reversed(range(ww - E)):
				for y in range(hh):
					if pic[y, x + E] == 0:
						pic[y, x + E] = pic[y, x]
						pic[y, x] = 0
		if S > 0:
			for y in reversed(range(hh - S)):
				for x in range(ww):
					if pic[y + S, x] == 0:
						pic[y + S, x] = pic[y, x]
						pic[y, x] = 0

		if W > 0:
			for x in range(ww - W):
				for y in range(hh):
					if pic[y, x] == 0:
						pic[y, x] = pic[y, x + W]
						pic[y, x + W] = 0

		return Block.new_picture(np_arr=pic)


	@bop_function(arg_types=[Block.type_picture], ret_type=Block.type_picture)
	def bopforward_pic_outline_4n(pic):
		"""
		Makes all pixels whose 4 neighbors (N,W,S,E) have the same color as them black.
		"""
		pic = pic.data.copy()

		hh, ww = pic.shape

		N = np.vstack([np.zeros(ww, dtype=np.int32), pic[0:hh-1, :]])
		W = np.hstack([pic[:, 1:ww], np.zeros((hh, 1), dtype=np.int32), ])
		S = np.vstack([pic[1:hh, :], np.zeros(ww, dtype=np.int32)])
		E = np.hstack([np.zeros((hh, 1), dtype=np.int32), pic[:, 0:ww-1]])

		mask = (pic > 0) * (N == pic) * (W == pic) * (S == pic) * (E == pic)

		pic[mask] = 0
		return Block.new_picture(np_arr=pic)


	@bop_function(arg_types=[Block.type_picture], ret_type=Block.type_picture)
	def bopforward_pic_rotate_90ccw(pic):
		"""
		Rotates the picture counterclockwise 90 degrees.
		"""
		return Block.new_picture(np_arr=np.rot90(pic.data))


	@bop_function(arg_types=[Block.type_picture], ret_type=Block.type_picture)
	def bopforward_pic_shape_on_auto_grid(pic):
		"""
		Returns a black picture with the shape taken from the cells inside an auto detected grid.
		"""
		hh, ww = pic.data.shape

		def auto_locate_grid_color():
			for xx in range(1, ww - 1):
				cc = int(pic.data[0, xx])
				if np.all(pic.data[:, xx] == cc):
					return cc
			for yy in range(1, hh - 1):
				cc = int(pic.data[yy, 0])
				if np.all(pic.data[yy, :] == cc):
					return cc
			return 0

		cc = auto_locate_grid_color()

		def next_black_row(oy):
			while oy < hh:
				if np.all(pic.data[oy, :] == cc):
					return oy
				oy += 1
			return oy

		def next_black_col(ox):
			while ox < ww:
				if np.all(pic.data[:, ox] == cc):
					return ox
				ox += 1
			return ox

		dy = 0
		oy = 0
		while oy < hh:
			ey = next_black_row(oy + 1)
			if ey - oy > 0:
				dy += 1
			oy = ey + 1

		dx = 0
		ox = 0
		while ox < ww:
			ex = next_black_col(ox + 1)
			if ex - ox > 0:
				dx += 1
			ox = ex + 1

		return Block.new_picture(np.zeros((dy, dx), dtype=np.int32))


	@bop_function(arg_types=[Block.type_picture], ret_type=Block.type_picture)
	def bopforward_pic_transpose(pic):
		"""
		Returns a picture transposed.
		"""
		return Block.new_picture(np_arr=pic.data.transpose())


	@bop_function(arg_types=[Block.type_picture], ret_type=Block.type_picture)
	def bopforward_pic_two_col_reverse(pic):
		"""
		Identifies the only two colors (hcf if more or less) and swaps them.
		"""
		pic = pic.data.copy()
		cc = np.unique(pic)

		if len(cc) != 2:
			return Block.new_error('Only two colors expected')

		mask = pic == cc[0]

		pic[mask] = cc[1]
		pic[np.logical_not(mask)] = cc[0]

		return Block.new_picture(np_arr=pic)


	@bop_function(arg_types=[Block.type_picture], ret_type=Block.type_picture)
	def bopforward_pic_v_axis(pic):
		"""
		Return a black picture with a 1 in the vertical axis (hcf when width is not odd) and the shape of pic.
		"""
		hh, ww = pic.data.shape

		if ww % 2 != 1:
			return Block.new_error('pic_v_axis() expects odd width')

		pic = np.zeros((hh, ww), dtype=np.int32)

		pic[:, int(ww/2)] = 1

		return Block.new_picture(np_arr=pic)


	@bop_function(arg_types=[Block.type_picture, Block.type_vector], ret_type=Block.type_picture)
	def bopforward_pic_vec_recolor_each(pic, vec):
		"""
		Returns a picture recolored pixel by pixel individually from the colors in a vector.
		"""
		pic = pic.data.copy()
		vv	= vec.data

		hh, ww = pic.shape
		N	   = len(vv)

		ii = 0
		for yy in range(hh):
			for xx in range(ww):
				pic[yy, xx] = vv[ii % N]

				ii += 1

		return Block.new_picture(np_arr=pic)


	@bop_function(arg_types=[Block.type_pictures], ret_type=Block.type_pictures)
	def bopforward_pics_filter_single_color(pics):
		"""
		Iterates over the pictures returning those with just one non-zero color.
		"""
		ret = ()
		for pic in pics.data:
			if len(np.unique(pic[pic != 0])) == 1:
				ret = ret + (pic,)

		if ret == ():
			return Block.new_error('Filter not found')

		return Block.new_pictures(ret)


	@bop_function(arg_types=[Block.type_pictures], ret_type=Block.type_picture)
	def bopforward_pics_filter_unique_pattern_as_pic(pics):
		"""
		Iterates over the pictures returning the only one that is unique in pattern or an error.
		"""
		kk = [str(((pic != 0)*1).tolist()) for pic in pics.data]

		patterns = dict((ky, kk.count(ky)) for ky in kk)

		ret = None
		for ky, pic in zip(kk, pics.data):
			if patterns[ky] == 1:
				if ret is not None:
					return Block.new_error('More than one unique pictures')
				ret = pic

		if ret is None:
			return Block.new_error('No unique pictures')

		return Block.new_picture(np_arr=ret)


	@bop_function(arg_types=[Block.type_pictures], ret_type=Block.type_picture)
	def bopforward_pics_filter_unique_picture_as_pic(pics):
		"""
		Iterates over the pictures returning the only one that is unique as a picture or an error.
		"""
		kk = [str(pic.tolist()) for pic in pics.data]

		pictures = dict((ky, kk.count(ky)) for ky in kk)

		ret = None
		for ky, pic in zip(kk, pics.data):
			if pictures[ky] == 1:
				if ret is not None:
					return Block.new_error('More than one unique pictures')
				ret = pic

		if ret is None:
			return Block.new_error('No unique pictures')

		return Block.new_picture(np_arr=ret)


	@bop_function(arg_types=[Block.type_pictures], ret_type=Block.type_pictures)
	def bopforward_pics_filter_v_symmetric(pics):
		"""
		Iterates over the pictures returning only those who are vertically symmetric.
		"""
		ret = ()
		for pic in pics.data:

			if np.array_equal(pic, np.fliplr(pic)):
				ret = ret + (pic,)

		if ret == ():
			return Block.new_error('Filter not found')

		return Block.new_pictures(ret)


	@bop_function(arg_types=[Block.type_pictures], ret_type=Block.type_vector)
	def bopforward_pics_main_color_as_vec(pics):
		"""
		Iterates over the pictures returning the main color of each as a vector.
		"""
		ret = []
		for pic in pics.data:
			cc = np.unique(pic, return_counts=True)
			ret.append(cc[0][np.argmax(cc[1])])

		return Block.new_vector(ret)


	@bop_function(arg_types=[Block.type_pictures], ret_type=Block.type_picture)
	def bopforward_pics_maximum_as_pic(pics):
		"""
		Iterates over pictures of the same shape computing the maximum for each pixel.
		"""
		if len(pics.data) == 0:
			return Block.new_error('Pics cannot be empty')

		hh, ww = pics.data[0].shape

		pic = np.zeros((hh, ww), dtype=np.int32)

		for ppp in pics.data:
			if ppp.data.shape != (hh, ww):
				return Block.new_error('All pics must have the same shape')

			pic = np.maximum(pic, ppp.data)

		return Block.new_picture(np_arr=pic)


	@bop_function(arg_types=[Block.type_pictures, Block.type_picture], ret_type=Block.type_picture)
	def bopforward_pics_pic_multiply_as_pic(bricks, layout):
		"""
		Returns an image repeating a list of automatically cropped bricks over each pixel of the layout.
		"""
		ey, ex = layout.data.shape

		pics = []

		for i, brick in enumerate(bricks.data):
			mask = brick != 0
			pic  = brick[np.ix_(mask.any(1), mask.any(0))]

			if i == 0:
				by, bx = pic.shape
			else:
				if (by, bx) != pic.shape:
					return Block.new_error('All bricks must have the same shape')

			pics.append(pic)

		N = ex*ey
		M = len(pics)
		if N % M != 0:
			return Block.new_error('Output cells not a multiple of bricks')

		ii = 0
		for yy in range(ey):
			for xx in range(ex):
				if xx == 0:
					row = pics[ii]
				else:
					row = np.hstack([row, pics[ii]])
				ii = (ii + 1) % M
			if yy == 0:
				pic = row
			else:
				pic = np.vstack([pic, row])

		return Block.new_picture(np_arr=pic)


	@bop_function(arg_types=[Block.type_vector], ret_type=Block.type_integer)
	def bopforward_vec_as_int(vv):
		"""
		Returns the only value in a vector as type_integer of hcf.
		"""
		if len(vv.data) != 1:
			return Block.new_error('Expects vector of length 1')

		return Block.new_integer(vv.data[0])


	@bop_function(arg_types=[Block.type_vector], ret_type=Block.type_integer)
	def bopforward_vec_length_as_int(vv):
		"""
		Returns the length of a vector.
		"""
		return Block.new_integer(len(vv.data))


	@bop_function(arg_types=[Block.type_vector], ret_type=Block.type_picture)
	def bopforward_vec_row_as_pic(vv):
		"""
		Returns a vector as a single row picture.
		"""
		return Block.new_picture(np_arr=np.array([vv.data], dtype=np.int32))
