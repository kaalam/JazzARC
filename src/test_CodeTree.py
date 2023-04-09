# Jazz (c) 2018-2023 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses

import math

from Block		import Block
from BopForward	import BopForward
from CodeTree	import CodeTree
from Example	import Example


def test_CodeTree():
	code_tree = CodeTree('./code_base/basis_code.jcb', Example)

	code_tree.build_opcodes_by_types()

	assert isinstance(code_tree.opcode_by_arg_type[Block.type_picture], dict)
	assert isinstance(code_tree.opcode_by_arg_type[Block.type_picture][Block.type_picture], dict)
	assert isinstance(code_tree.opcode_by_arg_type[Block.type_picture][Block.type_nothing], list)

	assert isinstance(code_tree.opcode_by_ret_type[Block.type_picture], list)
	assert Block.type_core not in code_tree.opcode_by_ret_type

	args_pic_nesw = code_tree.opcode_by_arg_type[Block.type_picture][Block.type_NESW][Block.type_nothing]

	assert BopForward.bopforward_pic_nesw_extend in args_pic_nesw

	code_tree.build_codeitems_from_codebase()
	code_tree.build_isomorphisms()

	nvs = 0
	nva = 0
	nci = 0
	np0 = 0
	npp = 0
	for use, variants in code_tree.variant_by_stackuse.items():
		nvs += 1
		for variant in variants:
			nva += 1
			for code_item in code_tree.alleles_by_variant[variant]:
				nci += 1

				assert code_item in code_tree.prior_by_codeitem

				if code_tree.prior_by_codeitem[code_item] == 0:
					np0 += 1
				else:
					npp += 1

	assert nvs >= 4 and nva > 250 and nci > 300 and np0 > 140 and npp > 150

	code_tree.aggregate_priors()

	sum_priors = 0
	for prior in code_tree.prior_by_codeitem.values():
		assert prior >= 1e-9 and prior <= 1
		sum_priors += prior

	sum_mp	   = 0
	items	   = 0
	last_prior = 0
	for use, moves in code_tree.item_prior_by_stackuse.items():
		nvs -= 1

		for i, (item, prior) in enumerate(moves):
			if i > 0:
				assert prior <= last_prior

			items	   += 1
			sum_mp	   += prior
			last_prior	= prior

	assert nvs == 0 and math.isclose(sum_mp, sum_priors) and items == npp + np0


def test_DebugUtils():
	code_tree = CodeTree('./code_base/basis_code.jcb', Example)

	hashes		 = set()
	code_by_hash = {}

	for use, variants in code_tree.variant_by_stackuse.items():
		for variant in variants:
			for code_item in code_tree.alleles_by_variant[variant]:
				hash = code_tree.hash_code(code_item)

				if hash in hashes:
					for a, b in zip(code_by_hash[hash], code_item):
						assert a.data == b.data

				hashes.add(hash)
				code_by_hash[hash] = code_item
