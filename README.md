# Hartree-Fock SCF Calculation for H₂ Molecule

A simple implementation of the Hartree-Fock Self-Consistent Field (SCF) method for calculating the electronic structure of the hydrogen molecule (H₂).

## Overview

This program performs a restricted Hartree-Fock calculation for the H₂ molecule using the STO-3G basis set at an internuclear distance of R = 1.4 bohr.

## Features

- **SCF Iteration**: Implements the self-consistent field procedure to solve the Hartree-Fock equations
- **Symmetric Orthogonalization**: Uses S^(-1/2) transformation for basis set orthogonalization
- **Proper ERI Handling**: Utilizes 8-fold symmetry for efficient storage and computation of electron repulsion integrals
- **Complete Energy Calculation**: Includes both electronic energy and nuclear-nuclear repulsion energy

## Theory

The Hartree-Fock method approximates the many-electron wavefunction as a single Slater determinant and solves the electronic structure problem iteratively:

1. **Fock Matrix Construction**: F_μν = H_μν^core + Σ_λσ P_λσ [(μν|λσ) - 0.5(μλ|νσ)]
2. **Orthogonalization**: Transform to orthogonal basis using X = S^(-1/2)
3. **Diagonalization**: Solve eigenvalue problem F'C' = ε C'
4. **Density Matrix Update**: P_μν = 2 Σ_i C_μi C_νi (sum over occupied orbitals)
5. **Energy Calculation**: E_total = E_elec + E_nuc

## Results

For H₂ at R = 1.4 bohr with STO-3G basis set:

```
Electronic Energy:    -1.8310386546 Hartree
Nuclear Repulsion:     0.7142857143 Hartree
Total Energy:         -1.1167529403 Hartree (-30.39 eV)

Orbital Energies:
  Orbital 1 (occupied):  -0.578221 Hartree
  Orbital 2 (virtual):    0.670489 Hartree
```

SCF converges in 2 iterations with an energy convergence threshold of 10⁻⁸ Hartree.

## Requirements

- Python 3.x
- NumPy

## Usage

```bash
python hartree_fock_h2.py
```

## Implementation Details

### Molecular Integrals

The program uses precomputed molecular integrals for H₂/STO-3G (R=1.4 bohr):

- **Overlap Matrix (S)**: Measures basis function overlap
- **Core Hamiltonian (H_core)**: Kinetic energy + nuclear-electron attraction
- **Electron Repulsion Integrals (ERI)**: Two-electron repulsion integrals with 8-fold symmetry

### Key Components

- `HartreeFock` class: Main SCF calculation engine
- `orthogonalize_basis()`: Symmetric orthogonalization using S^(-1/2)
- `build_fock_matrix()`: Constructs Fock matrix from density matrix
- `build_density_matrix()`: Builds density matrix from MO coefficients
- `scf_iteration()`: Main SCF loop with convergence checking

## Notes

- This is an educational implementation focusing on clarity rather than performance
- Real quantum chemistry calculations require integral evaluation routines
- The STO-3G basis set is minimal and provides only qualitative accuracy
- For production calculations, use established packages like PySCF, Psi4, or Gaussian

## References

- Szabo, A.; Ostlund, N. S. *Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory*
- Helgaker, T.; Jørgensen, P.; Olsen, J. *Molecular Electronic-Structure Theory*

## License

This code is provided for educational purposes.
