#!/usr/bin/env python3
"""
简单的Hartree-Fock SCF程序用于H2分子
假设所有单双电子积分已知
"""

import numpy as np

class HartreeFock:
    def __init__(self, S, H_core, eri, n_electrons, max_iter=100, convergence=1e-6):
        """
        初始化Hartree-Fock计算

        参数:
        S: 重叠积分矩阵 (n_basis x n_basis)
        H_core: 核心哈密顿矩阵 (单电子积分) (n_basis x n_basis)
        eri: 双电子排斥积分 (n_basis x n_basis x n_basis x n_basis)
        n_electrons: 电子数
        max_iter: 最大迭代次数
        convergence: 能量收敛阈值
        """
        self.S = S
        self.H_core = H_core
        self.eri = eri
        self.n_electrons = n_electrons
        self.n_occ = n_electrons // 2  # 占据轨道数(闭壳层)
        self.n_basis = S.shape[0]
        self.max_iter = max_iter
        self.convergence = convergence

        self.C = None  # 分子轨道系数
        self.P = None  # 密度矩阵
        self.E_total = 0.0
        self.orbital_energies = None

    def orthogonalize_basis(self):
        """
        使用对称正交化方法 (S^(-1/2))
        """
        # 对角化重叠矩阵
        s_eigval, s_eigvec = np.linalg.eigh(self.S)

        # 构建 S^(-1/2)
        s_diag_inv_sqrt = np.diag(1.0 / np.sqrt(s_eigval))
        self.X = s_eigvec @ s_diag_inv_sqrt @ s_eigvec.T

        return self.X

    def build_density_matrix(self, C):
        """
        构建密度矩阵 P
        P_μν = 2 * Σ_i C_μi * C_νi (对占据轨道求和)
        """
        P = np.zeros((self.n_basis, self.n_basis))
        for mu in range(self.n_basis):
            for nu in range(self.n_basis):
                for i in range(self.n_occ):
                    P[mu, nu] += 2.0 * C[mu, i] * C[nu, i]
        return P

    def build_fock_matrix(self, P):
        """
        构建Fock矩阵
        F_μν = H_μν^core + Σ_λσ P_λσ [(μν|λσ) - 0.5*(μλ|νσ)]
        """
        F = self.H_core.copy()

        for mu in range(self.n_basis):
            for nu in range(self.n_basis):
                for lam in range(self.n_basis):
                    for sigma in range(self.n_basis):
                        # Coulomb积分
                        J = self.eri[mu, nu, lam, sigma]
                        # 交换积分
                        K = self.eri[mu, lam, nu, sigma]

                        F[mu, nu] += P[lam, sigma] * (J - 0.5 * K)

        return F

    def calculate_electronic_energy(self, P, F):
        """
        计算电子能量
        E_elec = 0.5 * Σ_μν P_μν (H_μν^core + F_μν)
        """
        E_elec = 0.5 * np.sum(P * (self.H_core + F))
        return E_elec

    def scf_iteration(self):
        """
        执行SCF迭代
        """
        print("开始Hartree-Fock SCF计算...")
        print(f"基函数数量: {self.n_basis}")
        print(f"电子数: {self.n_electrons}")
        print(f"占据轨道数: {self.n_occ}")
        print("-" * 60)

        # 步骤1: 正交化基函数
        X = self.orthogonalize_basis()

        # 步骤2: 初始猜测 - 使用核心哈密顿矩阵
        F = self.H_core.copy()

        # 在正交化基组中对角化
        F_prime = X.T @ F @ X
        epsilon, C_prime = np.linalg.eigh(F_prime)

        # 转换回原始基组
        C = X @ C_prime

        # 构建初始密度矩阵
        P = self.build_density_matrix(C)

        # 计算初始能量
        E_old = self.calculate_electronic_energy(P, F)

        print(f"{'迭代':<8} {'电子能量':<20} {'能量变化':<20}")
        print("-" * 60)

        # SCF迭代
        for iteration in range(1, self.max_iter + 1):
            # 步骤3: 构建Fock矩阵
            F = self.build_fock_matrix(P)

            # 步骤4: 在正交化基组中求解本征方程
            F_prime = X.T @ F @ X
            epsilon, C_prime = np.linalg.eigh(F_prime)

            # 转换回原始基组
            C = X @ C_prime

            # 步骤5: 构建新的密度矩阵
            P_new = self.build_density_matrix(C)

            # 步骤6: 计算能量
            E_new = self.calculate_electronic_energy(P_new, F)

            # 计算能量变化
            delta_E = E_new - E_old

            print(f"{iteration:<8} {E_new:<20.10f} {delta_E:<20.10f}")

            # 检查收敛性
            if abs(delta_E) < self.convergence:
                print("-" * 60)
                print(f"SCF收敛! 迭代次数: {iteration}")
                self.C = C
                self.P = P_new
                self.E_total = E_new
                self.orbital_energies = epsilon
                return True

            # 更新密度矩阵和能量
            P = P_new
            E_old = E_new

        print("-" * 60)
        print(f"警告: SCF未在{self.max_iter}次迭代内收敛!")
        self.C = C
        self.P = P
        self.E_total = E_new
        self.orbital_energies = epsilon
        return False

    def print_results(self):
        """
        打印计算结果
        """
        print("\n" + "=" * 60)
        print("Hartree-Fock计算结果")
        print("=" * 60)
        print(f"总电子能量: {self.E_total:.10f} Hartree")
        print("\n轨道能量:")
        for i, eps in enumerate(self.orbital_energies):
            occ_str = "占据" if i < self.n_occ else "虚轨道"
            print(f"  轨道 {i+1}: {eps:12.6f} Hartree ({occ_str})")

        print("\n分子轨道系数:")
        print("(行: 基函数, 列: 分子轨道)")
        print(self.C)

        print("\n密度矩阵:")
        print(self.P)


def create_h2_test_integrals():
    """
    为H2分子创建测试积分
    使用STO-3G基组的近似值
    """
    # 2个基函数 (每个H原子一个)
    n_basis = 2

    # 重叠积分矩阵
    S = np.array([
        [1.0, 0.6593],
        [0.6593, 1.0]
    ])

    # 核心哈密顿矩阵 (动能 + 核吸引)
    H_core = np.array([
        [-1.1204, -0.9584],
        [-0.9584, -1.1204]
    ])

    # 双电子排斥积分 (μν|λσ)
    eri = np.zeros((n_basis, n_basis, n_basis, n_basis))

    # 填充双电子积分 (使用对称性)
    # (11|11)
    eri[0, 0, 0, 0] = 0.7746
    # (11|22)
    eri[0, 0, 1, 1] = 0.5697
    eri[1, 1, 0, 0] = 0.5697
    # (22|22)
    eri[1, 1, 1, 1] = 0.7746
    # (12|12)
    eri[0, 1, 0, 1] = 0.4441
    eri[1, 0, 1, 0] = 0.4441
    # (11|12)
    eri[0, 0, 0, 1] = 0.4441
    eri[0, 0, 1, 0] = 0.4441
    eri[0, 1, 0, 0] = 0.4441
    eri[1, 0, 0, 0] = 0.4441
    # (22|12)
    eri[1, 1, 0, 1] = 0.4441
    eri[1, 1, 1, 0] = 0.4441
    eri[0, 1, 1, 1] = 0.4441
    eri[1, 0, 1, 1] = 0.4441
    # (12|11)
    eri[0, 1, 0, 0] = 0.4441
    eri[1, 0, 0, 0] = 0.4441
    # (12|22)
    eri[0, 1, 1, 1] = 0.4441
    eri[1, 0, 1, 1] = 0.4441

    return S, H_core, eri


def main():
    """
    主函数 - 运行H2分子的Hartree-Fock计算
    """
    print("=" * 60)
    print("H2分子 Hartree-Fock计算程序")
    print("=" * 60)
    print()

    # 创建测试积分
    S, H_core, eri = create_h2_test_integrals()

    # H2分子有2个电子
    n_electrons = 2

    # 创建HF计算对象
    hf = HartreeFock(S, H_core, eri, n_electrons, max_iter=100, convergence=1e-8)

    # 执行SCF迭代
    converged = hf.scf_iteration()

    # 打印结果
    hf.print_results()

    print("\n注: 本程序使用的是H2分子STO-3G基组的近似积分值")
    print("实际计算中这些积分需要通过量子化学积分程序计算得到")


if __name__ == "__main__":
    main()
