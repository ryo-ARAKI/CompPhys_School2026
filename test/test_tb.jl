using LinearAlgebra
using Test
using GrapheneHHG

@testset "TB validity" begin
    tb0 = TBParams(; t=1.0, Δ=0.0)
    tbΔ = TBParams(; t=1.0, Δ=0.2)

    for k in kgrid_rhombus(6)
        Hk = H(k, tbΔ)
        # Hermite対称性を検証
        @test Hk ≈ Hk'
    end

    for k in kgrid_rhombus(6)
        vals = eigvals(Hermitian(Matrix(H(k, tb0))))
        # 二つの固有値が逆符号であることを検証
        @test vals[1] ≈ -vals[2]
    end

    K = GrapheneHHG.Vec2(2 * pi / (3 * sqrt(3)), 2 * pi / 3)
    valsK = eigvals(Hermitian(Matrix(H(K, tb0))))
    # 固有値が二つとも0であることを検証
    @test valsK[1] ≈ 0.0 atol=1e-12
    @test valsK[2] ≈ 0.0 atol=1e-12
end
