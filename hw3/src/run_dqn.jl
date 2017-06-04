using ReinforcementLearning, ArgParse, Knet, Logging

include("model.jl")
include("replay_buffer.jl")
include("utils.jl")

function dqn_learn(model, prms, env, buffer, exploration; args=nothing)
    state = getInitialState(env)
    total = 0.0
    readytosave = 100000
    episode_rewards = Float64[]
    frames = Float64[]

    tupdate = round(Int, args["frames"] / 20)
    target = copy(model)

    for fnum=1:args["frames"]
        if args["render"]
            render_env(env)
        end
        transformed = transform(state.data)
        stride = model["type"] == "mlp" ? nothing : args["stride"]
        if rand() < value(exploration, fnum) && !args["play"]
            a = ReinforcementLearning.sample(env).action + 1
        else
            obses_t = encode_recent(buffer, transformed; stack=args["stack"])
            inp = convert(args["atype"], obses_t)
            qvals = predict(model, inp; nh=model["nh"], stride=stride)
            a = indmax(Array(qvals))
        end
        selected = env.actions[a]
        state, reward = transfer(env, state, selected)
        total += reward 

        if !args["play"]
            transformed_n = transform(state.data)
            push!(buffer, transformed, a, reward, transformed_n, state.done)
            
            if can_sample(buffer, args["bs"])
                obses_t, actions, rewards, obses_tp1, dones = sample(buffer, args["bs"]; stack=args["stack"])
                obses_tp1 = convert(args["atype"], obses_tp1)
                nextq = predict(target, obses_tp1; nh=model["nh"], stride=stride)# use the target network to predict the q values
                nextq = Array(nextq)
                maxs = maximum(nextq,1)
                nextmax = sum(nextq .* (nextq.==maxs), 1)
                nextmax = reshape(nextmax, 1, length(nextmax))
                targets = reshape(rewards,1,length(rewards)) .+ (args["gamma"] .* nextmax .* dones)
                obses_t = convert(args["atype"], obses_t)
                targets = convert(args["atype"], targets)
                train!(model, prms, obses_t, actions, targets; nh=model["nh"], stride=stride)
            end

            if fnum > readytosave
                savemodel(model, args["save"])
                readytosave += 100000
            end

            if fnum % tupdate == 0
                target = copy(model)
            end
        end

        if isTerminal(state, env)
            state = getInitialState(env)
            info("Frame: $fnum , Total reward: $total, Exploration Rate: $(value(exploration, fnum))")
            push!(episode_rewards, total)
            push!(frames, fnum)
            total = 0.0
        end
    end
    return episode_rewards, frames
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
        ("--stride"; arg_type=Int; nargs='+'; default=nothing; help="strides used in conv")
        ("--env"; default="CartPole-v0")
        ("--render"; action=:store_true)
        ("--log"; default=""; help="log file")
        ("--level"; help = "log level"; default="INFO")
        ("--memory"; arg_type=Int; default=1000; help="memory size")
        ("--bs"; arg_type=Int; default=32; help="batch size")
        ("--stack"; arg_type=Int; default=4; help="length of the frame history")
        ("--save"; default="pong.jld"; help="model name")
        ("--load"; default=""; help="model name")
        ("--atype";default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"))
        ("--play"; action=:store_true; help="only play")
    end
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s)
    o["atype"] = eval(parse(o["atype"]))
    srand(1234)

    o["log"] != "" && Logging.configure(filename=o["log"])
    Logging.configure(level=eval(parse(o["level"])))

    for k in keys(o); info("$k => $(o[k])"); end
    env = GymEnv(o["env"])

    imgin = size(getInitialState(env))
    imgin = length(o["filters"]) != 0 ? (84, 84, o["stack"]) : (imgin[1]*o["stack"], imgin[2:end]...) #stack frames

    if o["load"] == ""
        model = init_weights(o["hiddens"], length(env.actions);
            windows=o["windows"], filters=o["filters"], stride=o["stride"],
            imgin=imgin, winit=o["winit"], atype=o["atype"])
    else
        model = loadmodel(o["load"], o["atype"])
    end

    prms = initparams(model;lr=o["lr"])

    buffer = ReplayBuffer(o["memory"])

    exploration = PiecewiseSchedule([(0, 1.0), (round(Int, o["frames"]/5), 0.1), (round(Int, o["frames"]/4), 0.01)])

    rewards, frames = dqn_learn(model, prms, env, buffer, exploration; args=o)
    save(string(split(o["save"], ".jld")[1], "_log.jld"), "rewards", rewards, "frames", frames)
end

main(ARGS)
