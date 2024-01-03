# Jazz (c) 2018-2024 kaalam.ai (The Authors of Jazz), released as:
#
#  1. You can use this for research under a GPL-3 license
#  2. See the main Jazz project or contact the authors (kaalam@kaalam.ai) for other licenses

"""
	This file contains global numerical constants only!!
	----------------------------------------------------

	When moved to Kagglespace, these constants will be replaced in the source code by their values.

	All configurable constants must be here and nothing else!!
"""

# MoveGen specific

MAX_MOVES_AT_ROOT		= 150
NUM_MOVES_STP_DISCOUNT	= 0.75

# nm <- MAX_MOVES_AT_ROOT
# st <- 1

# while (nm >= 1) {
# 	cat(sprintf('step: %3i, nm: %3i\n', st, nm))
# 	nm <- as.integer(nm*NUM_MOVES_STP_DISCOUNT)
# 	st <- st + 1
# }

# step:   1, nm: 150
# step:   2, nm: 112
# step:   3, nm:  84
# step:   4, nm:  63
# step:   5, nm:  47
# step:   6, nm:  35
# step:   7, nm:  26
# step:   8, nm:  19
# step:   9, nm:  14
# step:  10, nm:  10
# step:  11, nm:   7
# step:  12, nm:   5
# step:  13, nm:   3
# step:  14, nm:   2
# step:  15, nm:   1
# >

# Priors, rewards and evals

NUM_TOP_SOLUTIONS		= 3

WEIGHT_PRIOR_BY_FORM	= 0.3		# prior = (1 - WEIGHT_PRIOR_BY_FORM)*prior + WEIGHT_PRIOR_BY_FORM*prior_by_form[form]
WEIGHT_MIN_IN_EVAL		= 0.6		# ev = eval[IDX_PIC_REACH_MEAN]*(1 - WEIGHT_MIN_IN_EVAL) + eval[IDX_PIC_REACH_MIN]*WEIGHT_MIN_IN_EVAL
PRIOR_BOOST_IN_SNIPPET	= 0.5

# Constants for MCTS walks

ADD_EXP_NOISE_EACH		= 20
DIRICHLET_ALPHA 		= 0.1
EXPLORATION_FRAC		= 0.15

UCB_C_BASE				= 19652
UCB_C_INIT				= 1.25
UCB_DISCOUNT			= 1.2

REWARD_DISCOUNT			= 0.9

# Reward XGBoost

DUMMY_REWARD_INSTEAD	=  0
XGB_N_ESTIMATORS		= 25
XGB_ETA					= .3
XGB_GAMMA				=  0
XGB_MAX_DEPTH			=  6
XGB_MIN_CHILD_WEIGHT	=  1
XGB_MAX_DELTA_STEP		=  0
XGB_SUBSAMPLE			=  1
XGB_REG_LAMBDA			=  1
XGB_REG_ALPHA			=  0
XGB_VERBOSE				= -1

# Constants for special events in CodeEval

EVAL_FULL_MATCH			=  5
EVAL_WRONG_SHAPE		= -5
EVAL_MAX_PIC_SIZE		= 40

# Indices inside a 1x18 numpy array for eval

IDX_PIC_REACH_MIN		=  0
IDX_PIC_BETTER_MIN		=  1
IDX_PIC_WORSE_MIN		=  2
IDX_PAT_REACH_MIN		=  3
IDX_PAT_BETTER_MIN		=  4
IDX_PAT_WORSE_MIN		=  5

IDX_PIC_REACH_MEAN		=  6
IDX_PIC_BETTER_MEAN		=  7
IDX_PIC_WORSE_MEAN		=  8
IDX_PAT_REACH_MEAN		=  9
IDX_PAT_BETTER_MEAN		= 10
IDX_PAT_WORSE_MEAN		= 11

IDX_PIC_REACH_MAX		= 12
IDX_PIC_BETTER_MAX		= 13
IDX_PIC_WORSE_MAX		= 14
IDX_PAT_REACH_MAX		= 15
IDX_PAT_BETTER_MAX		= 16
IDX_PAT_WORSE_MAX		= 17

LENGTH_CODE_EVAL		= 18
