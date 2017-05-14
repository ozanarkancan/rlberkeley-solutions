using PyCall
importall ReinforcementLearning
import Base.==
import Base.hash
import Base.isequal

unshift!(PyVector(pyimport("sys")["path"]), "")

global const fl = PyCall.pywrap(PyCall.pyimport("frozen_lake"))

type FLState <: AbsState; data; done; end
type FLAction <: AbsAction; action; end

==(lhs::FLAction, rhs::FLAction) = lhs.action == rhs.action
isequal(lhs::FLAction, rhs::FLAction) = lhs.action == rhs.action
hash(a::FLAction) = hash(a.action)

==(lhs::FLState, rhs::FLState) = lhs.data == rhs.data
isequal(lhs::FLState, rhs::FLState) = lhs.data == rhs.data
hash(s::FLState) = hash(s.data)

type FrozenLake <: AbsEnvironment
    env
    states
    actions
end

function FrozenLake()
    env = fl.FrozenLakeEnv()
    states = [FLState(i, false) for i in 0:(env[:nS]-1)]
    actions = [FLAction(i) for i in 0:(env[:nA]-1)]
    return FrozenLake(env, states, actions)
end

getActions(s::FLState, env::FrozenLake) = env.actions
getInitialState(env::FrozenLake) = FLState(env.env[:reset](), false)
isTerminal(state::FLState, env::FrozenLake) = state.done

function transfer(env::FrozenLake, state::FLState, action::FLAction)
    obs, reward, done, info = env.env[:step](action.action)
    return (FLState(obs, done), reward)    
end

render_env(env::FrozenLake) = env.env[:render]()
sample(env::FrozenLake) = FLAction(env.env[:action_space][:sample]())

describe(env::FrozenLake) = env.env[:__doc__]
getAllStates(env::FrozenLake) = env.states

function getSuccessors(s::FLState, a::FLAction, env::FrozenLake)
    arr = env.env[:P][s.data][a.action]
    res = Array{Tuple{FLState, Float64, Float64}, 1}()

    for t in arr
        push!(res, (FLState(t[2], t[4]), t[3], t[1]))
    end

    return res
end
