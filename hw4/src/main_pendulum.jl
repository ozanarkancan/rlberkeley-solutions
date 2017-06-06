importall ReinforcementLearning
using Knet, ArgParse, Logging, JLD

type ReinforceAgent <: AbsAgent
	weights
	prms
    steplimit
	states
	actions
	rewards
    step
	γ
end

function ReinforceAgent(env::GymEnv; α=0.001, γ=0.99, steplimit=20, hid=100)
	w = Dict()
	D = length(getInitialState(env).data)
	O = 1

    m = randn(hid, D)
    norm = sum(sqrt(m .* m))
	w["wh"] = convert(Array{Float32}, (m*0.1) ./ norm)
	w["bh"] = zeros(Float32, hid, 1)
   
    m = randn(O, hid)
    norm = sum(sqrt(m .* m))
	w["w"] = convert(Array{Float32}, (m*0.1) ./ norm)
	w["b"] = zeros(Float32, O, 1)

	prms = Dict()
    for k in keys(w); prms[k] = Adam(lr=α); end

	ReinforceAgent(w, prms; γ=γ, steplimit=steplimit)
end

ReinforceAgent(w, prms; γ=0.99, steplimit=20) = ReinforceAgent(w, prms, steplimit, Any[], Any[], Array{Float64, 1}(), 0, γ)

function lrelu(x, leak=0.2)
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x+ f2 *abs(x)
end

function predict(wh, bh, w, b, x)
    h = lrelu(wh * x .+ bh)
	mu = w * h .+ b
end

function sample_action(mu; sigma=1.0)
    a = mu + randn() * sigma
end

function logpdf(mu,x; sigma=1.0)    
    fac = -log(sqrt(2pi))
    r = (x-mu)/ sigma
    return -r.* r.* 0.5 - log(sigma) + fac
end

function loss(w, x, actions, rewards)
	mus = predict(w["wh"], w["bh"], w["w"], w["b"], x)
    nll = sum(-(logpdf(mus, actions) .* rewards) ./ size(mus, 2))
end

lossgradient = grad(loss)

function discount(rewards; γ=0.9)
	discounted = zeros(Float32, 1, length(rewards))
	discounted[end] = rewards[end]

	for i=(length(rewards)-1):-1:1
		discounted[i] = rewards[i] + γ * discounted[i+1]
	end
	return discounted
end

function play(agent::ReinforceAgent, state::GymState, env::GymEnv; learn=false)

	mu = predict(agent.weights["wh"], agent.weights["bh"], agent.weights["w"], agent.weights["b"], state.data)
	a = sample_action(mu)

	if learn
        s = convert(Array{Float32}, reshape(state.data, length(state.data), 1))
        act = convert(Array{Float32}, reshape(a, length(a), 1))

        if length(agent.states) == 0
            agent.states = s
            agent.actions = act
        else
            agent.states = hcat(agent.states, s)
            agent.actions = hcat(agent.actions, act)
        end
	end
    a = reshape(a, 1,)
	return GymAction(a)
end


function observe(agent::ReinforceAgent, state::AbsState, reward::Float64, env::AbsEnvironment; learn=true, terminal=false)
	if learn
        agent.step += 1
		push!(agent.rewards, reward)
		if terminal && agent.step > agent.steplimit
			disc_rewards = discount(agent.rewards; γ=agent.γ)
            disc_rewards -= mean(disc_rewards) #standardize the rewards to be unit normal
            disc_rewards ./ std(disc_rewards)
            
            g = lossgradient(agent.weights, agent.states, agent.actions, disc_rewards)

			for k in keys(g)
				update!(agent.weights[k], g[k], agent.prms[k])
			end

			agent.states = Any[]
			agent.actions = Any[]
			agent.rewards = Array{Float64, 1}()
            agent.step = 0
		end
	end
end

function main(args=ARGS)
	s = ArgParseSettings()
	s.description="Solution with REINFORCE algorithm"
	s.exc_handler=ArgParse.debug_handler
	@add_arg_table s begin
		("--lr"; arg_type=Float64; default=0.01; help="learning rate")
		("--gamma"; arg_type=Float64; default=0.9; help="discount rate")
		("--render"; action=:store_true; help="display the env")
		("--epoch"; arg_type=Int; default=200; help="number of epochs")
		("--hidden"; arg_type=Int; default=32; help="number of units in the hidden layer")
		("--steplimit"; arg_type=Int; default=10; help="number of steps to train the agent")
		("--seed"; arg_type=Int; default=123; help="random seed")
		("--threshold"; arg_type=Int; default=10000; help="stop the episode even it is not terminal after number of steps exceeds the threshold")
        ("--log"; default="")
	end
	o = parse_args(args, s)
	srand(o["seed"])
    o["log"] != "" && Logging.configure(filename=o["log"])
    Logging.configure(level=INFO)
	env = GymEnv("Pendulum-v0")
	agent = ReinforceAgent(env; α=o["lr"], γ=o["gamma"], steplimit=o["steplimit"], hid=o["hidden"])
	rewards = Array{Float64, 1}()

    totalsteps = 0.0
    
    steps = Float64[]
    means = Float64[]

	for i=1:o["epoch"]
		totalRewards, numberOfStates = playEpisode(env, agent; learn=true, threshold = o["threshold"], render=o["render"], verbose=false)
        totalsteps += numberOfStates
        push!(rewards, totalRewards)
		msg = string("Episode ", i, " , total rewards: ", totalRewards, " , steps: ", totalsteps)
		if i >= 100
			m = mean(rewards[(i-100+1):end])
			msg = string(msg, " , mean: ", m)
            push!(means, m)
            push!(steps, totalsteps)
		end
		info(msg)
	end
    save(string(split(o["log"],".log")[1], "_experiment.jld"), "rewards", means, "steps", steps)
end

main()
