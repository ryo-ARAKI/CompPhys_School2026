using LinearAlgebra
using Test
using GrapheneHHG

@testset "GKSL properties" begin
    Nk = 4
    kgrid = kgrid_rhombus(Nk)
    tb = TBParams(; t=1.0, Δ=0.1)
    pulse = default_pulse(; A0=0.3, ω0=0.7, fwhm_cycles=5.0)

    cache = precompute_lindblad(kgrid, tb)
    ρ = ground_state_density(kgrid, tb)
    dρ = zero_state(length(kgrid))

    p = RHSParams(; kgrid=kgrid, tb=tb, pulse=pulse, lindblad=cache, γ=0.1)
    rhs!(dρ, ρ, p, 0.0)

    for i in eachindex(dρ)
        # トレースの保存を検証
        @test tr(dρ[i]) ≈ 0.0 atol = 1e-12
        # Hermite対称性を検証
        @test dρ[i] ≈ dρ[i]'
    end

    for LdL in cache.LdL
        # L^\dagger Lが射影演算子であることを検証
        # Hermite対称性を検証
        @test LdL ≈ LdL'
        # トレースが1であることを検証
        @test tr(LdL) ≈ 1.0
        # 冪等性を検証
        @test LdL * LdL ≈ LdL
    end
end
