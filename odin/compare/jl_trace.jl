#!/usr/bin/env julia
# Machine-checkable execution trace for cross-validating the Odin port.
# Mirrors the plain (uncoupled) run_simulation loop exactly:
#   init_state -> per MCS: mcs_step! + update_melanin! + update_nutrient!
#   + update_centers_of_mass!
# Prints TRACE/TRACECELL/TRACESPEC lines; the Odin binary's --trace flag
# prints byte-identical lines for diffing.

using Printf, SHA

function load_serial()
    src = read(joinpath(@__DIR__, "..", "..", "biofilms_potts.jl"), String)
    src = split(src, "#  13. Figure export")[1]
    M = Module(:SerialRef)
    Base.eval(M, :(using LinearAlgebra, Statistics, Random, Printf))
    Base.include_string(M, src, "biofilms_potts.jl")
    return M
end

function dump_state(SR, state, seed, m)
    lat = state.lattice
    lath = bytes2hex(sha256(vec(reinterpret(UInt8, lat))))
    melh = bytes2hex(sha256(vec(reinterpret(UInt8, state.melanin))))
    nuth = bytes2hex(sha256(vec(reinterpret(UInt8, state.nutrient))))
    radh = bytes2hex(sha256(vec(reinterpret(UInt8, state.radiation))))
    println("TRACE $seed $m LAT $lath MEL $melh NUT $nuth RAD $radh ALIVE $(length(state.cells))")
    for id in sort!(collect(keys(state.cells)))
        c = state.cells[id]
        println("TRACECELL $seed $m $id $(c.species) $(c.volume) " *
                "$(reinterpret(UInt64, c.com[1])) $(reinterpret(UInt64, c.com[2])) " *
                "$(reinterpret(UInt64, c.com[3]))")
    end
    snap = SR.take_snapshot(state, m)
    for sd in snap.species_data
        println("TRACESPEC $seed $m $(sd.species) $(sd.total_volume) $(sd.n_cells) " *
                "$(reinterpret(UInt64, sd.mean_r)) $(reinterpret(UInt64, sd.mean_melanin))")
    end
end

function dump_rd(SR, rd, contaminant, seed, m)
    ch = bytes2hex(sha256(vec(reinterpret(UInt8, rd.c))))
    sh = bytes2hex(sha256(vec(reinterpret(UInt8, rd.s))))
    kh = bytes2hex(sha256(vec(reinterpret(UInt8, contaminant))))
    println("RDC $seed $m $(reinterpret(UInt64, rd.t)) $(reinterpret(UInt64, rd.m)) $ch $sh $kh")
end

function main_trace(SR, N, cells, n_mcs, seed, coupled)
    params = SR.CPMParams(N = N, n_cells_per_species = cells,
                          snapshot_interval = 100000)
    rng = SR.MersenneTwister(seed)
    state = SR.init_state(params; seed = seed)
    local rd = nothing
    local contaminant = nothing
    if coupled
        rp = SR.RadiolysisParams()
        rd = SR.init_radiolysis(rp; R = Float64(N) / 2.0)
        contaminant = zeros(Float64, N, N, N)
    end
    SR.update_centers_of_mass!(state) # run_simulation does this before snap 0
    dump_state(SR, state, seed, 0)
    for m in 1:n_mcs
        SR.mcs_step!(state, rng)
        if coupled
            if m % 10 == 1
                X_tot, X_rd = SR.compute_radial_biomass(state, rd.params.Nr)
                rd.params = SR.RadiolysisParams(
                    rd.params.Nr, rd.params.D_eff, rd.params.k_ads, rd.params.k_red,
                    rd.params.k_des, rd.params.k_loss, SR.mean(X_tot), SR.mean(X_rd),
                    rd.params.P0, rd.params.alpha_P, rd.params.k_dam,
                    rd.params.Ddot_R, rd.params.c_ext, rd.params.dt_rd)            end
            SR.step_radiolysis!(rd, rd.params.dt_rd)
            SR.radial_to_3d!(contaminant, rd.c, rd.r_grid, state.interior, N)
            SR.update_melanin!(state)
            SR.update_nutrient_coupled!(state, rd.m)
        else
            SR.update_melanin!(state)
            SR.update_nutrient!(state)
        end
        SR.update_centers_of_mass!(state)
        dump_state(SR, state, seed, m)
        coupled && dump_rd(SR, rd, contaminant, seed, m)
    end
end

function _cli()
    N       = parse(Int, get(ENV, "TRACE_N", "16"))
    cells   = parse(Int, get(ENV, "TRACE_CELLS", "2"))
    n_mcs   = parse(Int, get(ENV, "TRACE_MCS", "8"))
    seed    = parse(Int, get(ENV, "TRACE_SEED", "42"))
    coupled = get(ENV, "TRACE_COUPLED", "0") == "1"
    Base.invokelatest(main_trace, load_serial(), N, cells, n_mcs, seed, coupled)
end

_cli()
