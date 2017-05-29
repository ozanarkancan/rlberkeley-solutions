using ReinforcementLearning, ArgParse, Knet

include("model.jl")
include("replay_buffer.jl")
include("utils.jl")

function dqn_learn(model, prms, env, buffer, exploration; args=nothing)
    state = getInitialState(env)
    total = 0.0

    for fnum=1:args["frames"]
        if args["render"]
            render_env(env)
        end
        transformed = transform(state.data)
        stride = model["type"] == "mlp" ? nothing : args["stride"]
        if rand() < value(exploration, fnum)
            a = ReinforcementLearning.sample(env).action + 1
        else
            qvals = predict(model, transformed; nh=model["nh"], stride=stride)
            a = indmax(qvals)
        end
        selected = env.actions[a]
        state, reward = transfer(env, state, selected)
        transformed_n = transform(state.data)
        push!(buffer, transformed, a, reward, transformed_n, state.done)
        total += reward
        if isTerminal(state, env)
            state = getInitialState(env)
            println("Frame: $fnum , Total reward: $total, Exploration Rate: $(value(exploration, fnum))")
            total = 0.0
        end
        if can_sample(buffer, args["bs"])
            obses_t, actions, rewards, obses_tp1, dones = sample(buffer, args["bs"])
            nextq = predict(model, obses_tp1; nh=model["nh"], stride=stride)
            maxs = maximum(nextq,1)
            nextmax = sum(nextq .* (nextq.==maxs), 1)
            nextmax = reshape(nextmax, 1, length(nextmax))
            targets = reshape(rewards,1,length(rewards)) .+ args["gamma"] .* nextmax .* dones
            train!(model, prms, obses_t, actions, targets; nh=model["nh"], stride=stride)
        end
    end
end

function main(args=ARGS)
    s = ArgParseSettings()
    s.description = "Q Learning"
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--frames"; arg_type=Int; default=100; help="number of frames")
        ("--winit"; arg_type=Float64; default=0.01; help="winit")
        ("--lr"; arg_type=Float64; default=0.001; help="learning rate")
        ("--gamma"; arg_type=Float64; default=0.99; help="discount factor")
        ("--gclip"; arg_type=Float64; default=5.0; help="threshold for the gradient clipping")
        ("--hiddens"; arg_type=Int; nargs='+'; default=[32]; help="number of units in the hiddens for the mlp")
        ("--filters"; arg_type=Int; nargs='+'; default=nothing; help="number of filters at each layer")
        ("--windows"; arg_type=Int; nargs='+'; default=nothing; help="window size")
        ("--stride"; arg_type=Int; nargs='+'; default=[4]; help="strides used in conv")
        ("--env"; default="CartPole-v0")
        ("--render"; action=:store_true)
        ("--log"; default=""; help="log file")
        ("--level"; help = "log level"; default="INFO")
        ("--memory"; arg_type=Int; default=1000; help="memory size")
        ("--bs"; arg_type=Int; default=32; help="batch size")
    end
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s)

    for k in keys(o); println("$k => $(o[k])"); end
    env = GymEnv(o["env"])

    imgin = size(getInitialState(env))
    imgin = o["filters"] != nothing ? (84, 84, 1) : imgin

    model = init_weights(o["hiddens"], length(env.actions); windows=o["windows"], filters=o["filters"], stride=o["stride"], imgin=imgin, winit=o["winit"], atype=Array{Float32})
    prms = initparams(model;lr=o["lr"])

    buffer = ReplayBuffer(o["memory"])

    exploration = PiecewiseSchedule([(0, 1.0), (100000, 0.1), (round(Int, o["frames"]/4), 0.01)])

    dqn_learn(model, prms, env, buffer, exploration; args=o)
end

main(ARGS)
