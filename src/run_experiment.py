from Search import Search


stop_rlz = {
	'broken_threshold'		: 0.08,
	'max_broken_walks'		: 50,
	'max_elapsed_sec'		: 12,
	'min_num_walks'			: 50,
	'stop_num_full_matches' : 1
}


x_path = Search.build_experiment(100, 100, stop_rlz)
# x_path = './experiments/2020-05-24_180322'

print(x_path)

res = Search.run_experiment(x_path)

print('\n'.join(res))
