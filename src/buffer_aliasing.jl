@inline _index_sets_overlap(i::Integer, j::Integer) = i == j
@inline _index_sets_overlap(i::Integer, r::AbstractRange) = i in r
@inline _index_sets_overlap(r::AbstractRange, j::Integer) = j in r
@inline _index_sets_overlap(r1::AbstractRange, r2::AbstractRange) = !isempty(intersect(r1, r2))
@inline _index_sets_overlap(_, _) = true

function _unwrap_nonview_parent(A::AbstractArray)
    curr = A
    while !(curr isa SubArray)
        p = parent(curr)
        p === curr && return curr
        curr = p
    end
    return curr
end

function _arrays_alias(a::AbstractArray, b::AbstractArray)
    (isempty(a) || isempty(b)) && return false
    ua = _unwrap_nonview_parent(a)
    ub = _unwrap_nonview_parent(b)
    ua === ub && return true
    ra = ua isa SubArray ? _unwrap_nonview_parent(parent(ua)) : ua
    rb = ub isa SubArray ? _unwrap_nonview_parent(parent(ub)) : ub
    if (ra isa SubArray) || (rb isa SubArray)
        while parent(ra) !== ra
            ra = parent(ra)
        end
        while parent(rb) !== rb
            rb = parent(rb)
        end
        return ra === rb
    end
    ra === rb || return false
    ia = ua isa SubArray ? parentindices(ua) : axes(ra)
    ib = ub isa SubArray ? parentindices(ub) : axes(rb)
    length(ia) == length(ib) || return true
    for d in eachindex(ia, ib)
        _index_sets_overlap(ia[d], ib[d]) || return false
    end
    return true
end

function _disjoint_hemispheric_buffers(outputs, inputs)
    for (i, a) in enumerate(outputs)
        any(b -> _arrays_alias(a, b), inputs) &&
            throw(ArgumentError("Output buffer aliases source input"))
        any(b -> _arrays_alias(a, b), outputs[1:(i - 1)]) &&
            throw(ArgumentError("Output buffers alias each other"))
    end
    return nothing
end
