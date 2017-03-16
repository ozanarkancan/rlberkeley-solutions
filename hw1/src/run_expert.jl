using ReinforcementLearning, ArgParse, JLD

include("load_policy.jl")

function parse_commandline()
	s = ArgParseSettings()

	@add_arg_table s begin
		("--envname"; help = "environment")
		("--render"; action = :store_true)
		("--max_timesteps"; default=0; arg_type=Int)
		("--num_rollouts"; arg_type=Int; default=20; help="Number of expert roll outs")
	end
	return parse_args(s)
end

function main()
	args = parse_commandline()
	println("Loading expert policy")

	policy, ws = load_policy(string("experts/", args["envname"], ".pkl"))
	env = GymEnv(args["envname"])

	returns = Any[]
	observations = Any[]
	actions = Any[]

	max_steps = args["max_timesteps"] != 0 ? args["max_timesteps"] : env.spec.timestep_limit
    println("Nondeterministic: $(env.spec.nondeterministic)")

	for i=1:args["num_rollouts"]
		obs = getInitialState(env)
		totalr = 0.0
		num_of_states = 0.0

		while !isTerminal(obs, env) && num_of_states <= max_steps
			a = policy(ws, convert(Array{Float32}, reshape(obs.data, 1, size(obs.data, 1))))
			push!(observations, obs)
			push!(actions, a)
			obs, r = transfer(env, obs, GymAction(a))
			totalr += r
			num_of_states += 1
			if args["render"]
				render_env(env)
			end
		end

        println("Epoch: $i , Total Reward: $totalr")
	end

    println("Saving...")
    save(string("experts/", args["envname"], ".jld"), "obs", observations, "acts", actions)
end

main()
