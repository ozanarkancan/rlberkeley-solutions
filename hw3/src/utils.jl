using Images

"""
Resize the image applying the low pass filter first.
Reference: http://juliaimages.github.io/latest/function_reference.html#Spatial-transformations-and-resizing-1
"""
function transform(img, sz)
    if length(size(img)) < 3
        return reshape(img, size(img)..., 1)
    end

    σ = map((o,n)->0.75*o/n, size(img), sz)
    kern = KernelFactors.gaussian(σ)
    imgr = imresize(imfilter(img, kern, NA()), sz)
    return reshape(imgr, size(imgr)...,1)
end

linear_interpolation(l, r, alpha) = l + alpha * (r - l)

abstract AbsSchedule
value(schedule::AbsSchedule, t) = error("Not implemented")

type PiecewiseSchedule <: AbsSchedule
    endpoints
    interpolation
end

PiecewiseSchedule(endpoints) = PiecewiseSchedule(endpoints, linear_interpolation)

function value(schedule::PiecewiseSchedule, t)
    for ((l_t, l),(r_t,r)) in zip(schedule.endpoints[1:end-1],schedule.endpoints[2:end])
        if l_t <= t && t < r_t
            alpha = (t - l_t) / (r_t - l_t)
            return schedule.interpolation(l, r, alpha)
        end
    end
    return schedule.endpoints[end][2]
end
