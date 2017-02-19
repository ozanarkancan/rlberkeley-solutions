using PyCall
@pyimport cPickle as pickle

function lrelu(x; leak=0.2)
	f1 = 0.5 * (1 + leak)
	f2 = 0.5 * (1 - leak)

	return f1 * x + f2 * abs(x)
end

read_layer(l) = (convert(Array{Float32}, l["AffineLayer"]["W"]), convert(Array{Float32}, l["AffineLayer"]["b"]))

function load_policy(filename)
	f = open(filename)
	data = pickle.load(PyTextIO(f))
	close(f)

	nonlin_type = data["nonlin_type"]
	policy_type = [k for k in keys(data) if k != "nonlin_type"][1]
	@assert policy_type == "GaussianPolicy"

	policy_params = data[policy_type]
	obsnorm_mean = policy_params["obsnorm"]["Standardizer"]["mean_1_D"]
	obsnorm_meansq = policy_params["obsnorm"]["Standardizer"]["meansq_1_D"]
	obsnorm_stdev = sqrt(max(0, obsnorm_meansq - (obsnorm_mean .^ 2)))

	layer_params = policy_params["hidden"]["FeedforwardNet"]
	ws = Any[]
	for layer_name in sort(collect(keys(layer_params)))
		l = layer_params[layer_name]
		W, b = read_layer(l)
		push!(ws, W)
		push!(ws, b)
		println("Layer: $(layer_name), $(size(W)), $(size(b))")
	end
	W, b = read_layer(policy_params["out"])
	push!(ws, W)
	push!(ws, b)
	println("Layer: out, $(size(W)), $(size(b))")
	
	function policy(ws, obs_bo; μ=obsnorm_mean, μ2=obsnorm_meansq, σ=obsnorm_stdev, nonlin=nonlin_type)
		#println("Input: $(size(obs_bo))")
		normedobs_bo = (obs_bo .- obsnorm_mean) ./ (obsnorm_stdev + 1e-6)
		curr_activations_bd = normedobs_bo

		for i=1:2:(length(ws)-2)
			raw = curr_activations_bd * ws[i] .+ ws[i+1]
			if nonlin == "lrelu"
				curr_activations_bd = lrelu(raw; leak=.01)
			elseif nonlin == "tanh"
				curr_activations_bd = tanh(raw)
			end
		end
		raw = curr_activations_bd * ws[end-1] .+ ws[end]
		#println("Out: $(size(raw))")
		return raw
	end

	return policy, ws
end


