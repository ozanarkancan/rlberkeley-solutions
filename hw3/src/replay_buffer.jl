import Base.length
import Base.push!

"""
Replay buffer implementation based on https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
"""
type ReplayBuffer
    size
    storage
    next_idx
end

"""
Constructor
"""
ReplayBuffer(size) = ReplayBuffer(size, [], 1)

"""
Length of the buffer
"""
length(buf::ReplayBuffer) = length(buf.storage)

"""
Tells whether there is enough data in the buffer or not
"""
can_sample(buf::ReplayBuffer, batch_size) = batch_size <= length(buf)

"""
Add new transition to the buffer
"""
function push!(buf::ReplayBuffer, obs_t, action, reward, obs_tp1, done)
    data = (obs_t, action, reward, obs_tp1, done)

    if buf.next_idx > length(buf)
        push!(buf.storage, data)
    else
        buf.storage[buf.next_idx] = data
    end

    buf.next_idx = max(1, (buf.next_idx + 1) % (buf.size+1))
end

function encode_sample(buf::ReplayBuffer, idxes)
    bs = length(idxes)
    obses_t = zeros(Float32, size(buf.storage[1][1])[1:(end-1)]..., bs)
    actions = zeros(Int, bs)
    rewards = zeros(Float32, bs)
    obses_tp1 = zeros(Float32, size(buf.storage[1][1])[1:(end-1)]..., bs)
    dones = zeros(Float32, 1, bs)

    indx = 0
    for ind in idxes
        indx += 1
        data = buf.storage[ind]
        obs_t, action, reward, obs_tp1, done = data
        obses_t[map(t->1:t, size(obs_t)[1:(end-1)])..., indx] = obs_t
        actions[indx] = action
        rewards[indx] = reward
        obses_tp1[map(t->1:t, size(obs_t)[1:(end-1)])..., indx] = obs_tp1
        dones[1, indx] = done ? 0.0 : 1.0
    end
    obses_t, actions, rewards, obses_tp1, dones
end

function sample(buf::ReplayBuffer, batchsize)
    idxes = randperm(length(buf))[1:batchsize]
    return encode_sample(buf, idxes)
end
