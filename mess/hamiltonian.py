# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnl
import optimistix as optx
from jaxtyping import Array

from mess.basis import Basis
from mess.integrals import eri_basis, kinetic_basis, nuclear_basis, overlap_basis
from mess.interop import to_pyscf
from mess.mesh import Mesh, density, density_and_grad, xcmesh_from_pyscf
from mess.scf import otransform_symmetric
from mess.structure import nuclear_energy
from mess.types import FloatNxN, OrthTransform
from mess.xcfunctional import (
    lda_correlation_vwn,
    lda_exchange,
    gga_correlation_pbe,
    gga_exchange_pbe,
    gga_exchange_b88,
    gga_correlation_lyp,
)


class TwoElectron(eqx.Module):
    eri: Array

    def __init__(self, basis: Basis, backend: str = "mess"):
        """

        Args:
            basis (Basis): the basis set used to build the electron repulsion integrals
            backend (str, optional): Integral backend used. Defaults to "mess".
        """
        super().__init__()
        if backend == "mess":
            self.eri = eri_basis(basis)
        elif backend == "pyscf":
            mol = to_pyscf(basis.structure, basis.basis_name)
            self.eri = jnp.array(mol.intor("int2e_cart", aosym="s1"))

    def coloumb(self, P: FloatNxN) -> FloatNxN:
        """Build the Coloumb matrix (classical electrostatic) from the density matrix.

        Args:
            P (FloatNxN): the density matrix

        Returns:
            FloatNxN: Coloumb matrix
        """
        return jnp.einsum("kl,ijkl->ij", P, self.eri)

    def exchange(self, P: FloatNxN) -> FloatNxN:
        """Build the quantum-mechanical exchange matrix from the density matrix

        Args:
            P (FloatNxN): the density matrix

        Returns:
            FloatNxN: Exchange matrix
        """
        return jnp.einsum("ij,ikjl->kl", P, self.eri)


class HartreeFockExchange(eqx.Module):
    two_electron: TwoElectron

    def __init__(self, two_electron: TwoElectron):
        self.two_electron = two_electron

    def __call__(self, P: FloatNxN) -> float:
        K = self.two_electron.exchange(P)
        return -0.25 * jnp.sum(P * K)


class LDA(eqx.Module):
    basis: Basis
    mesh: Mesh

    def __init__(self, basis: Basis):
        self.basis = basis
        self.mesh = xcmesh_from_pyscf(basis.structure)

    def __call__(self, P: FloatNxN) -> float:
        rho = density(self.basis, self.mesh, P)
        eps_xc = lda_exchange(rho) + lda_correlation_vwn(rho)
        E_xc = jnp.einsum("i,i,i", self.mesh.weights, rho, eps_xc)
        return E_xc


class PBE(eqx.Module):
    basis: Basis
    mesh: Mesh

    def __init__(self, basis: Basis):
        self.basis = basis
        self.mesh = xcmesh_from_pyscf(basis.structure)

    def __call__(self, P: FloatNxN) -> float:
        rho, grad_rho = density_and_grad(self.basis, self.mesh, P)
        eps_xc = gga_exchange_pbe(rho, grad_rho) + gga_correlation_pbe(rho, grad_rho)
        E_xc = jnp.einsum("i,i,i", self.mesh.weights, rho, eps_xc)
        return E_xc


class PBE0(eqx.Module):
    basis: Basis
    mesh: Mesh
    hfx: HartreeFockExchange

    def __init__(self, basis: Basis, two_electron: TwoElectron):
        self.basis = basis
        self.mesh = xcmesh_from_pyscf(basis.structure)
        self.hfx = HartreeFockExchange(two_electron)

    def __call__(self, P: FloatNxN) -> float:
        rho, grad_rho = density_and_grad(self.basis, self.mesh, P)
        e = 0.75 * gga_exchange_pbe(rho, grad_rho) + gga_correlation_pbe(rho, grad_rho)
        E_xc = jnp.einsum("i,i,i", self.mesh.weights, rho, e)
        return E_xc + 0.25 * self.hfx(P)


class B3LYP(eqx.Module):
    basis: Basis
    mesh: Mesh
    hfx: HartreeFockExchange

    def __init__(self, basis: Basis, two_electron: TwoElectron):
        self.basis = basis
        self.mesh = xcmesh_from_pyscf(basis.structure)
        self.hfx = HartreeFockExchange(two_electron)

    def __call__(self, P: FloatNxN) -> float:
        rho, grad_rho = density_and_grad(self.basis, self.mesh, P)
        eps_x = 0.08 * lda_exchange(rho) + 0.72 * gga_exchange_b88(rho, grad_rho)
        vwn_c = (1 - 0.81) * lda_correlation_vwn(rho)
        lyp_c = 0.81 * gga_correlation_lyp(rho, grad_rho)
        b3lyp_xc = eps_x + vwn_c + lyp_c
        E_xc = jnp.einsum("i,i,i", self.mesh.weights, rho, b3lyp_xc)
        return E_xc + 0.2 * self.hfx(P)


def build_xcfunc(xc_method: str, basis: Basis, two_electron: TwoElectron) -> eqx.Module:
    if xc_method == "lda":
        return LDA(basis)
    if xc_method == "pbe":
        return PBE(basis)
    if xc_method == "pbe0":
        return PBE0(basis, two_electron)
    if xc_method == "b3lyp":
        return B3LYP(basis, two_electron)
    if xc_method == "hfx":
        return HartreeFockExchange(two_electron)

    raise ValueError(f"Unsupported exchange-correlation option: {xc_method}")


class Hamiltonian(eqx.Module):
    X: FloatNxN
    H_core: FloatNxN
    basis: Basis
    two_electron: TwoElectron
    xcfunc: eqx.Module

    def __init__(
        self,
        basis: Basis,
        otransform: OrthTransform = otransform_symmetric,
        xc_method: str = "lda",
    ):
        super().__init__()
        S = overlap_basis(basis)
        self.X = otransform(S)
        self.H_core = kinetic_basis(basis) + nuclear_basis(basis).sum(axis=0)
        self.basis = basis
        self.two_electron = TwoElectron(basis, backend="pyscf")
        self.xcfunc = build_xcfunc(xc_method, basis, self.two_electron)

    def __call__(self, P: FloatNxN) -> float:
        E_core = jnp.sum(self.H_core * P)
        E_xc = self.xcfunc(P)
        J = self.two_electron.coloumb(P)
        E_es = 0.5 * jnp.sum(J * P)
        E = E_core + E_xc + E_es
        return E


@jax.jit
def minimise(H: Hamiltonian):
    def f(Z, _):
        C = H.X @ jnl.qr(Z).Q
        P = H.basis.density_matrix(C)
        return H(P)

    solver = optx.BFGS(rtol=1e-5, atol=1e-6)
    Z = jnp.eye(H.basis.num_orbitals)
    sol = optx.minimise(f, solver, Z)
    C = H.X @ jnl.qr(sol.value).Q
    P = H.basis.density_matrix(C)
    E_elec = H(P)
    E_total = E_elec + nuclear_energy(H.basis.structure)
    return E_total, C, sol
