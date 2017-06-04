function mlp(w, x; nh=1)
    inp = x
    for i=1:nh
        inp = relu(w[string("w_",i)] * inp .+ w[string("b_", i)])
    end
    q = w["w_out"] * inp .+ w["b_out"]
    return q
end

function convnet(w, x, stride; nh=2)
    inp = x
    for i=1:nh
        inp = relu(conv4(w[string("w_",i)], inp; stride=stride[i]) .+ w[string("b_", i)])
    end
    inp = reshape(inp, prod(size(inp)[1:3]), size(inp)[end])
    h = relu(w["fully_w"] * inp .+ w["fully_b"])
    q = w["w_out"] * h .+ w["b_out"]
    return q
end

function predict(w, x; nh=1, stride=nothing)
    if w["type"] == "mlp"
        return mlp(w, x; nh=nh)
    else
        return convnet(w, x, stride; nh=nh)
    end
end

function loss(model, states, actions, targets; nh=1, stride=nothing, val=[])
    qvals = predict(model, states; nh=nh, stride=stride)
    nrows = size(qvals, 1)
    index = actions + nrows*(0:(length(actions)-1))
    qpred = reshape(qvals[index], size(targets)...)
    mse = sumabs2(targets-qpred) / size(states, 2)
    push!(val, AutoGrad.getval(mse))
    return mse
end

lossgradient = grad(loss)

function train!(model, prms, states, actions, targets; nh=1, stride=nothing)
    val = Float64[0.0]
    g = lossgradient(model, states, actions, targets; nh=nh, stride=stride)
    update!(model, g, prms)
    return val[1]
end

function init_weights(hiddens, nout; windows=nothing, filters=nothing, stride=nothing, imgin=(32,32,3), winit=0.1, atype=Array{Float32})
    w = Dict{String, Any}()
    if length(filters) == 0
        w["type"] = "mlp"
        w["nh"] = length(hiddens)
        inp = prod(imgin)
        for i=1:length(hiddens)
            w[string("w_",i)] = winit * randn(hiddens[i], inp)
            w[string("b_",i)] = zeros(hiddens[i], 1)
            inp = hiddens[i]
        end
    else
        w["type"] = "convnet"
        w["nh"] = length(filters)
        inp = [imgin...]
        for i=1:length(filters)
            w[string("w_",i)] = winit * randn(windows[i], windows[i], inp[3], filters[i])
            w[string("b_",i)] = zeros(1, 1, filters[i], 1)
            inp[1] = floor((inp[1] - windows[i]) / stride[i]) + 1
            inp[2] = floor((inp[2] - windows[i]) / stride[i]) + 1
            inp[3] = filters[i]
        end
        inp = prod(inp)
        w["fully_w"] = winit * randn(hiddens[1], inp)
        w["fully_b"] = zeros(hiddens[1], 1)
        inp = hiddens[1]
    end
    w["w_out"] = winit * randn(nout, inp)
    w["b_out"] = zeros(nout, 1)
    for k in keys(w); if !(k=="type" || k=="nh"); w[k] = convert(atype, w[k]); end; end;
    return w
end

function initparams(w; lr=0.001)
    d = Dict()
    for k in keys(w)
        d[k] = Adam(lr=lr)
    end
    return d
end

function savemodel(w, fname)
    d = Dict()
    for k in keys(w); if !(k=="type" || k=="nh"); d[k] = convert(Array{Float32}, w[k]); end; end;
    d["type"] = w["type"]
    d["nh"] = w["nh"]
    save(fname, "model", d)
end

function loadmodel(fname, atype)
    w = load(fname, "model")
    for k in keys(w); if !(k=="type" || k=="nh"); w[k] = convert(atype, w[k]); end; end;
    return w
end
