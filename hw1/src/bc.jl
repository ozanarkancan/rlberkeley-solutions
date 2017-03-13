using ReinforcementLearning, ArgParse, JLD, DataFrames, Knet

include("load_policy.jl")

function parse_commandline()
	s = ArgParseSettings()

	@add_arg_table s begin
		("--envname"; help = "environment")
		("--render"; action = :store_true)
		("--max_timesteps"; default=0; arg_type=Int)
		("--num_rollouts"; arg_type=Int; default=20; help="Number of expert roll outs")
        ("--hidden"; nargs='*'; arg_type=Int; help="sizes of hidden layers, e.g. --hidden 128 64 for a net with two hidden layers")
        ("--epoch"; default=10; arg_type=Int)
        ("--bs"; default=64; arg_type=Int)
        ("--opt"; default="Adam"; help="Optimization method")
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
	end
	return parse_args(s)
end

function weights(inp, outpt, h...; atype=Array{Float32}, winit=0.1)
    w = Any[]
    x = inp
    for y in [h..., outpt]
        push!(w, convert(atype, winit*randn(x, y)))
        push!(w, convert(atype, zeros(1, y)))
        x = y
    end
    return w
end

function bc_policy(w, x)
    for i=1:2:length(w)
        x = x*w[i] .+ w[i+1]
        if i<length(w)-1
            x = relu(x) # max(0,x)
        end
    end
    return x
end

function loss(w, x, ygold)
    ypred = bc_policy(w, x)
    return sumabs2(ygold - ypred) / size(x, 1)
end

lossgradient = grad(loss)

function rollouts(policy, ws, envname; max_timesteps=0, numrollouts=100, render=false)
	env = GymEnv(envname)

	returns = Float32[]

	max_steps = max_timesteps != 0 ? max_timesteps : env.spec.timestep_limit

    atype = typeof(ws[1])

	for i=1:numrollouts
		obs = getInitialState(env)
		totalr = 0.0
		num_of_states = 0.0

		while !isTerminal(obs, env) && num_of_states <= max_steps
			a = policy(ws, convert(atype, reshape(obs.data, 1, size(obs.data, 1))))
			obs, r = transfer(env, obs, GymAction(convert(Array{Float32}, a)))
			totalr += r
			num_of_states += 1
			if render
				render_env(env)
			end
		end

        println("Epoch: $i , Total Reward: $totalr")
        push!(returns, totalr)
	end
    return returns
end

function minibatch(obs, acts, inps, outs; bs=10, atype=Array{Float32})
    data = Any[]

    for i=1:bs:length(obs)
        bl = i+bs-1 > length(obs) ? length(obs) : i+bs-1
        X = zeros(Float32, bl-i+1, inps)
        Y = zeros(Float32, bl-i+1, outs)
        for j=1:(bl-i+1)
            X[j, :] = obs[i+j-1].data
            Y[j, :] = acts[i+j-1]
        end
        push!(data, (convert(atype, X), convert(atype, Y)))
    end
    return data
end

function train(w, data, opts)
    for (x, y) in data
        g = lossgradient(w, x, y)
        update!(w, g, opts)
    end
    
    lss = 0.0
    count = 0.0

    for (x,y) in data
        lss += loss(w, x, y) * size(x, 1)
        count += size(x, 1)
    end
    return (lss / count)
end

function train_bc()
    args = parse_commandline()
	println("Loading expert policy")

	policy, ws = load_policy(string("experts/", args["envname"], ".pkl"))
    inps = size(ws[1], 1)
    outs = size(ws[end-1], 2)

    w = weights(inps, outs, args["hidden"]...; atype=eval(parse(args["atype"])))
    optf = eval(parse(args["opt"]))
    opts = map(x->optf(), w)
    d = load(string("experts/", args["envname"], ".jld"))
    observations = d["obs"]
    actions = d["acts"]

    data = minibatch(observations, actions, inps, outs; bs=args["bs"], atype=eval(parse(args["atype"])))

    for i=1:args["epoch"]
        lss = train(w, data, opts)
        println("Epoch: $i , Loss: $lss")
    end
    returns_expert = rollouts(policy, ws, args["envname"]; max_timesteps=args["max_timesteps"], numrollouts=10, render=false)
    returns_bc = rollouts(bc_policy, w, args["envname"]; max_timesteps=args["max_timesteps"], numrollouts=10, render=false)

    println("Expert: mean: $(mean(returns_expert)) std: $(std(returns_expert))")
    println("Bc: mean: $(mean(returns_bc)) std: $(std(returns_bc))")
end

train_bc()

#=
*** STABLE MODELS ***
Humanoid: --hidden 512 512 512 --bs 512 --epoch 400
=#
