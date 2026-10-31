from dataclasses import dataclass
from functools import cache
from math import comb, gamma, pi, sqrt

import numpy as np
from scipy.special import hyp1f1
from scipy.linalg.blas import get_blas_funcs

from pyscf import gto



ALL_INT2E_TYPES = ('bbbb',
                   'kbbb',
                   'kkbb',
                   'kbbk',
                   'kbkb',
                   'kkkb',
                   'kkkk')

@dataclass
class INT2e:
    index: tuple[int, int, int, int]
    value: complex
    # 'bbbb', 'kbbb', 'kbbk', 'kkbb', and so on
    type: str                                                                   

# Integral types use chemist notation (i,j|k,l). In the two-body excitation, i and k are creation-side indices, j and l are annihilation-side indices. And general symmetry for complex ERI: ijkl = klij = jilk* = lkji*.
#For mix system: b = bound AO/MO (ijkl), k = plane wave (pqrs)
# 0pw4ao types (8-fold symmetry):
## bbbb: abcd = bacd = abdc = badc = cdab = dcab = cdba = dcba (b->b/b->b); only store abcd with a >= b, c >= d, ab >= cd
# 1pw3ao types (4-fold symmetry):
## kbbb: pabc = bcpa = apcb* = cbap* (k->b/b->b or b->k/b->b); only store pabc with b >= c
# 2pw2ao types (4-fold symmetry):
## kkbb: pqab = pqba = qpba* = qpab* (k->k/b->b); only store pqab with p >= q, a >= b
## kbbk: pabq = bqpa = apqb* = qbap* (k->b/b->k); only store pabq with mixed pair (p, a) >= (q, b)
## kbkb: paqb = qbpa = apbq* = bqap* (k->b/k->b or b->k/b->k); only store paqb with mixed pair (p, a) >= (q, b)
# 3pw1ao types (4-fold symmetry):
## kkkb: pqra = rapq = qpar* = arqp* (k->k/k->b or k->k/b->k); only store fixed orientation pqra, with no p/q/r ordering
# 4pw0ao types (4-fold symmetry):
## kkkk: pqrs = rspq = qpsr* = srqp* (k->k/k->k); only store a canonical representative among these four tuples


def _hermite_coulomb(tx: int, ty: int, tz: int, order: int, alpha: float, 
                     pc: np.ndarray, boys_arg: np.ndarray) -> np.ndarray:
    '''
    Calculate the Hermite Coulomb integral.
    Args:
        tx, ty, tz: Cartesian powers.
        order: Order of the Hermite Coulomb integral.
        alpha: Exponent of the Gaussian function.
        pc: Array of complex centers.
        boys_arg: Argument of the Boys function.
    Returns:
        Value of the Hermite Coulomb integral.
    '''
    def _boys(order: int, arg: np.ndarray) -> np.ndarray:
        '''
        Calculate the Boys function.
        Args:
            order: Order of the Boys function.
            arg: Argument of the Boys function.
        Returns:
            Value of the Boys function.
        '''
        value = hyp1f1(order + 0.5, order + 1.5, -arg) / (2 * order + 1)
        bad = ~np.isfinite(value)
        if np.any(bad):
            arg = np.asarray(arg, dtype=complex)
            value = np.array(value, copy=True)
            if np.any(np.real(arg[bad]) <= 50.0):
                raise FloatingPointError(f"Boys function failed for order={order} and argument={arg[bad][:4]}")
            value[bad] = 0.5 * gamma(order + 0.5) * arg[bad] ** (-(order + 0.5))

        return value

    @cache
    def recurse(tx: int, ty: int, tz: int, order: int) -> np.ndarray:
        if tx == 0 and ty == 0 and tz == 0:
            return (-2.0 * alpha) ** order * _boys(order, boys_arg)
        if tx > 0:
            value = pc[:, 0] * recurse(tx - 1, ty, tz, order + 1)
            if tx > 1:
                value += (tx - 1) * recurse(tx - 2, ty, tz, order + 1)
            return value
        if ty > 0:
            value = pc[:, 1] * recurse(tx, ty - 1, tz, order + 1)
            if ty > 1:
                value += (ty - 1) * recurse(tx, ty - 2, tz, order + 1)
            return value

        value = pc[:, 2] * recurse(tx, ty, tz - 1, order + 1)
        if tz > 1:
            value += (tz - 1) * recurse(tx, ty, tz - 2, order + 1)
        return value

    return recurse(tx, ty, tz, order)


def _cartesian_powers(angular: int) -> list[tuple[int, int, int]]:
    '''
    Generate a list of Cartesian powers for a given angular momentum.
    Args:
        angular: Angular momentum.
    Returns:
        List of Cartesian powers.
    '''
    return [
        (power_x, power_y, angular - power_x - power_y)
        for power_x in range(angular, -1, -1)
        for power_y in range(angular - power_x, -1, -1)
    ]


def _central_hermite_coeff(power: int, alpha: float) -> np.ndarray:
    coeffs = np.zeros(power + 1, dtype=complex)
    coeffs[0] = 1.0
    max_order = 0
    for _ in range(power):
        next_coeffs = np.zeros(power + 1, dtype=complex)
        for order in range(max_order + 1):
            value = coeffs[order]
            next_coeffs[order + 1] += value / (2.0 * alpha)
            if order > 0:
                next_coeffs[order - 1] += order * value
        coeffs = next_coeffs
        max_order += 1

    return coeffs


def _shifted_hermite_coeff(power: int, shift: np.ndarray, alpha: float) -> np.ndarray:
    coeffs = np.zeros((power + 1,) + shift.shape, dtype=complex)
    for centered_power in range(power + 1):
        shifted_coeff = comb(power, centered_power) * shift ** (power - centered_power)
        hermite_coeffs = _central_hermite_coeff(centered_power, alpha)
        for order, coeff in enumerate(hermite_coeffs):
            coeffs[order] += coeff * shifted_coeff

    return coeffs


def _pw_pair_transfer(pw_k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta_k = pw_k[None, :, :] - pw_k[:, None, :]
    delta_k2 = np.einsum("pqi,pqi->pq", delta_k, delta_k)

    return delta_k, delta_k2


def ao_overlap(mol: gto.Mole) -> np.ndarray:
    '''Calculate AO overlap with the same coefficients and spherical transform as the AO-PW integrals.'''
    def _primitive_overlap(left_exponent: float, left_center: np.ndarray,
                           left_powers: tuple[int, int, int], right_exponent: float,
                           right_center: np.ndarray, right_powers: tuple[int, int, int]) -> float:
        '''
        Imam:
        Gaussian product rule, see Section 9.2.3 in Helgaker et al.
        '''
        exponent_sum = left_exponent + right_exponent
        product_center = (left_exponent * left_center + right_exponent * right_center) / exponent_sum
        displacement = left_center - right_center
        value = np.exp(-left_exponent * right_exponent / exponent_sum * np.dot(displacement, displacement))

        for axis in range(3):
            moments = np.zeros(left_powers[axis] + right_powers[axis] + 1)
            moments[0] = sqrt(pi / exponent_sum)
            for degree in range(2, len(moments), 2):
                moments[degree] = (degree - 1) * moments[degree - 2] / (2.0 * exponent_sum)

            axis_overlap = 0.0
            for left_degree in range(left_powers[axis] + 1):
                left_factor = (comb(left_powers[axis], left_degree)
                               * (product_center[axis] - left_center[axis]) ** (left_powers[axis] - left_degree))
                for right_degree in range(right_powers[axis] + 1):
                    axis_overlap += (left_factor * comb(right_powers[axis], right_degree)
                                     * (product_center[axis] - right_center[axis]) ** (right_powers[axis] - right_degree)
                                     * moments[left_degree + right_degree])
            value *= axis_overlap

        return float(value)

    ao_loc = mol.ao_loc_nr()
    shells = []
    for shell_id in range(mol.nbas):
        angular = mol.bas_angular(shell_id)
        exponents = np.asarray(mol.bas_exp(shell_id), dtype=float)
        coefficients = np.asarray(mol.bas_ctr_coeff(shell_id), dtype=float)
        coefficients *= gto.gto_norm(angular, exponents)[:, None]
        shells.append((
            np.asarray(mol.atom_coord(mol.bas_atom(shell_id)), dtype=float),
            exponents, coefficients, _cartesian_powers(angular),
            np.asarray(gto.cart2sph(angular), dtype=float),
        ))
        #^^ imam: note that the argument normalized ='sp' is not used in the call to gto.cart2sph above
        # (and in the other places across this file). This means that the s and p cartesian AOs to be
        # transformed should not normalize to unity. Beware that this behavior is in contrast to the
        # default choice of the same argument in mole.cart2sph_coeff.

    overlap = np.zeros((mol.nao_nr(), mol.nao_nr()))
    for left_shell_id, (left_center, left_exponents, left_coefficients, left_powers, left_transform) in enumerate(shells):
        left_start, left_stop = int(ao_loc[left_shell_id]), int(ao_loc[left_shell_id + 1])
        for right_shell_id in range(left_shell_id + 1):
            right_center, right_exponents, right_coefficients, right_powers, right_transform = shells[right_shell_id]
            right_start, right_stop = int(ao_loc[right_shell_id]), int(ao_loc[right_shell_id + 1])
            shell_cart = np.zeros((len(left_powers), left_coefficients.shape[1],
                                   len(right_powers), right_coefficients.shape[1]))
            for left_prim_id, left_exponent in enumerate(left_exponents):
                for right_prim_id, right_exponent in enumerate(right_exponents):
                    pair_coefficients = np.outer(left_coefficients[left_prim_id],
                                                 right_coefficients[right_prim_id])
                    for left_cart_id, left_power in enumerate(left_powers):
                        for right_cart_id, right_power in enumerate(right_powers):
                            primitive = _primitive_overlap(left_exponent, left_center, left_power,
                                                           right_exponent, right_center, right_power)
                            shell_cart[left_cart_id, :, right_cart_id, :] += primitive * pair_coefficients

            shell_sph = np.einsum("as,atbu,br->tsur", left_transform, shell_cart, right_transform,
                                  optimize=True).reshape(left_stop - left_start, right_stop - right_start)
            overlap[left_start:left_stop, right_start:right_stop] = shell_sph
            if left_shell_id != right_shell_id:
                overlap[right_start:right_stop, left_start:left_stop] = shell_sph.T

    return overlap


def _to_mol_shell_order(angular: int, shell_cart: np.ndarray) -> np.ndarray:
    '''
    Convert Cartesian shell values to molecular shell order.
    Args:
        angular: Angular momentum.
        shell_cart: Cartesian shell values.
    Returns:
        Shell values in molecular shell order.
    '''
    cart2sph = np.asarray(gto.cart2sph(angular), dtype=float)
    shell_values = np.einsum("cs,cnp->nsp", cart2sph, shell_cart)

    return shell_values.reshape(-1, shell_cart.shape[2])


def primitive_pw_moment(alpha: float, center: np.ndarray, powers: tuple[int, int, int], 
                         pw_k: np.ndarray) -> np.ndarray:
    '''
    Calculate the primitive plane wave moment.
    Args:
        alpha: Exponent of the Gaussian function.
        center: Center of the Gaussian function.
        powers: Cartesian powers.
        pw_k: Plane wave vectors.
    Returns:
        Value of the primitive plane wave moment.
    '''
    def _moment_1d(power: int, k: np.ndarray, alpha: float) -> np.ndarray:
        moments = [sqrt(pi / alpha) * np.exp(-(k * k) / (4.0 * alpha)) + 0.j]
        if power == 0:
            return moments[0]

        moments.append(1j * k * moments[0] / (2.0 * alpha))
        for order in range(1, power):
            moments.append((order * moments[order - 1] + 1j * k * moments[order]) / (2.0 * alpha))

        #vv imam: It only returns the last element of moments, it would be more efficient
        #         if it returns all elements already calculated before arriving at the power-th
        #         element.
        return moments[power]

    phase = np.exp(1j * np.einsum("pi,i->p", pw_k, center))
    moment_x = _moment_1d(powers[0], pw_k[:, 0], alpha)
    moment_y = _moment_1d(powers[1], pw_k[:, 1], alpha)
    moment_z = _moment_1d(powers[2], pw_k[:, 2], alpha)

    return phase * moment_x * moment_y * moment_z


def ao_pw_ovlp(mol: gto.Mole, pw_k: np.ndarray, pw_norm: float = 1.0) -> np.ndarray:

    '''
    Calculate the overlap integral of the plane wave with the Gaussian basis functions.
    Args:
        mol: PySCF molecule object.
        pw_k: Plane wave vectors.
        pw_norm: Normalization factor for the plane waves.
    Returns:
        Overlap integral matrix.
    '''
    num_pw = pw_k.shape[0]
    ao_pw = np.zeros((mol.nao_nr(), num_pw), dtype=complex)
    ao_loc = mol.ao_loc_nr()

    #vv imam: Loop over contractions (shells).
    for shell_id in range(mol.nbas):
        angular = mol.bas_angular(shell_id)
        atom_id = mol.bas_atom(shell_id)
        center = np.asarray(mol.atom_coord(atom_id), dtype=float)
        exponents = np.asarray(mol.bas_exp(shell_id), dtype=float)
        coefficients = np.asarray(mol.bas_ctr_coeff(shell_id), dtype=float)
        #^^imam: not sure why need np.asarray, mol.bas_ctr_coeff returns an array already.
        coefficients *= gto.gto_norm(angular, exponents)[:, None]
    
        powers_list = _cartesian_powers(angular)
        shell_cart = np.zeros((len(powers_list), coefficients.shape[1], num_pw), dtype=complex)
        #imam print('l = ', angular)
        #imam print('  powers = ', powers_list)
        #imam print("  shell_cart.shape = ", shell_cart.shape)
        #vv imam: Loop over primitive Gaussians making up the current contraction.
        for prim_id, alpha in enumerate(exponents):
            for cart_id, powers in enumerate(powers_list):
                prim = primitive_pw_moment(alpha, center, powers, pw_k)
                shell_cart[cart_id] += coefficients[prim_id, :, None] * prim[None, :]
                #^^imam: coefficients[prim_id, :] and prim[:] are vectors and we want to get their outer product.
                #        It could also be achieved with with
                #        shell_cart[cart_id] += np.outer(coefficients[prim_id, :], prim[:]).
                #print('a, b ', coefficients[prim_id, :, None].shape, prim[None, :].shape)
    
        start, stop = int(ao_loc[shell_id]), int(ao_loc[shell_id + 1])
        ao_pw[start:stop] = _to_mol_shell_order(angular, shell_cart)
    
    return pw_norm * ao_pw



def ao_pw_pot(mol: gto.Mole, pw_k: np.ndarray, pw_norm: float = 1.0) -> np.ndarray:
    def _primitive_nuclear(alpha: float, center: np.ndarray, powers: tuple[int, int, int],
                           charge: float, nuc_center: np.ndarray) -> np.ndarray:
        '''
        Calculate the primitive nuclear integral.
        Args:
            alpha: Exponent of the Gaussian function.
            center: Center of the Gaussian function.
            powers: Angular momentum powers (lx, ly, lz).
            charge: Nuclear charge.
            nuc_center: Center of the nucleus.
        Returns:
            Primitive nuclear integral.
        '''
        k2 = np.einsum("pi,pi->p", pw_k, pw_k)
        complex_center = center[None, :] + 1j * pw_k / (2.0 * alpha)
        center_shift = complex_center - center[None, :]
        pc = complex_center - nuc_center[None, :]
        boys_arg = alpha * np.einsum("pi,pi->p", pc, pc)
        phase = np.exp(1j * np.einsum("pi,i->p", pw_k, center) - k2 / (4.0 * alpha))

        coeff_x = _shifted_hermite_coeff(powers[0], center_shift[:, 0], alpha)
        coeff_y = _shifted_hermite_coeff(powers[1], center_shift[:, 1], alpha)
        coeff_z = _shifted_hermite_coeff(powers[2], center_shift[:, 2], alpha)
        prefactor = 2.0 * pi / alpha

        value = np.zeros(pw_k.shape[0], dtype=complex)
        for tx, cx in enumerate(coeff_x):
            for ty, cy in enumerate(coeff_y):
                coeff_xy = cx * cy
                for tz, cz in enumerate(coeff_z):
                    value += coeff_xy * cz * prefactor * _hermite_coulomb(tx, ty, tz, 0, alpha, pc, boys_arg)

        return -charge * phase * value

    num_pw = pw_k.shape[0]
    ao_pw = np.zeros((mol.nao_nr(), num_pw), dtype=complex)
    ao_loc = mol.ao_loc_nr()
    atom_coords = [np.asarray(mol.atom_coord(i), dtype=float) for i in range(mol.natm)]
    atom_charges = [float(mol.atom_charge(i)) for i in range(mol.natm)]
    
    for shell_id in range(mol.nbas):
        angular = mol.bas_angular(shell_id)
        atom_id = mol.bas_atom(shell_id)
        center = np.asarray(mol.atom_coord(atom_id), dtype=float)
        exponents = np.asarray(mol.bas_exp(shell_id), dtype=float)
        coefficients = np.asarray(mol.bas_ctr_coeff(shell_id), dtype=float)
        coefficients *= gto.gto_norm(angular, exponents)[:, None]
    
        powers_list = _cartesian_powers(angular)
        shell_cart = np.zeros((len(powers_list), coefficients.shape[1], num_pw), dtype=complex)
        for prim_id, alpha in enumerate(exponents):
            for cart_id, powers in enumerate(powers_list):
                prim = np.zeros(num_pw, dtype=complex)
                for nuc_id in range(mol.natm):
                    prim += _primitive_nuclear(alpha, center, powers, atom_charges[nuc_id],
                                              atom_coords[nuc_id])
                shell_cart[cart_id] += coefficients[prim_id, :, None] * prim[None, :]
    
        start, stop = int(ao_loc[shell_id]), int(ao_loc[shell_id + 1])
        ao_pw[start:stop] = _to_mol_shell_order(angular, shell_cart)
    
    return pw_norm * ao_pw
        

def ao_pw_dipole(mol: gto.Mole, pw_k: np.ndarray, pw_norm: float = 1.0) -> np.ndarray:
    num_pw = pw_k.shape[0]
    ao_pw = np.zeros((3, mol.nao_nr(), num_pw), dtype=complex)
    ao_loc = mol.ao_loc_nr()

    for shell_id in range(mol.nbas):
        angular = mol.bas_angular(shell_id)
        atom_id = mol.bas_atom(shell_id)
        center = np.asarray(mol.atom_coord(atom_id), dtype=float)
        exponents = np.asarray(mol.bas_exp(shell_id), dtype=float)
        coefficients = np.asarray(mol.bas_ctr_coeff(shell_id), dtype=float)
        coefficients *= gto.gto_norm(angular, exponents)[:, None]

        powers_list = _cartesian_powers(angular)
        shell_cart = np.zeros((3, len(powers_list), coefficients.shape[1], num_pw), dtype=complex)
        for prim_id, alpha in enumerate(exponents):
            for cart_id, powers in enumerate(powers_list):
                base = primitive_pw_moment(alpha, center, powers, pw_k)
                shifted_powers = (
                    (powers[0] + 1, powers[1], powers[2]),
                    (powers[0], powers[1] + 1, powers[2]),
                    (powers[0], powers[1], powers[2] + 1),
                )
                for axis, axis_powers in enumerate(shifted_powers):
                    prim = primitive_pw_moment(alpha, center, axis_powers, pw_k)
                    prim += center[axis] * base
                    shell_cart[axis, cart_id] += coefficients[prim_id, :, None] * prim[None, :]

        start, stop = int(ao_loc[shell_id]), int(ao_loc[shell_id + 1])
        for axis in range(3):
            ao_pw[axis, start:stop] = _to_mol_shell_order(angular, shell_cart[axis])
    
    return pw_norm * ao_pw


def int_1e(mol: gto.Mole, pw_k: np.ndarray, pw_norm: float = 1.0, op: str = "i") -> np.ndarray:
    '''
    calculate the integral of the plane wave with the Gaussian basis functions
    op = "i" for overlap
    op = "kin" for kinetic energy operator
    op = "pot" for potential energy operator
    op = "r" for dipole operator
    '''
    num_ao = mol.nao_nr()
    num_pw, _ = pw_k.shape
    num_basis = num_ao + num_pw

    def _int_ovlp() -> np.ndarray:
        ovlp = np.zeros((num_basis, num_basis), dtype=complex)
    
        ovlp[:num_ao, :num_ao] = mol.intor('int1e_ovlp_sph', hermi=1)

        ovlp[num_ao:, num_ao:] = np.eye(num_pw)
    
        ao_pw = ao_pw_ovlp(mol, pw_k, pw_norm)
        ovlp[:num_ao, num_ao:] = ao_pw
        ovlp[num_ao:, :num_ao] = ovlp[:num_ao, num_ao:].conj().T
    
        return ovlp

    def _int_kin() -> np.ndarray:
        kin = np.zeros((num_basis, num_basis), dtype=complex)

        kin[:num_ao, :num_ao] = mol.intor('int1e_kin_sph', hermi=1)

        pw_kin = 0.5 * np.einsum("pi,pi->p", pw_k, pw_k)
        kin[num_ao:, num_ao:] = np.diag(pw_kin)

        ao_pw_overlap = ao_pw_ovlp(mol, pw_k, pw_norm)
        kin[:num_ao, num_ao:] = ao_pw_overlap * pw_kin[None, :]
        kin[num_ao:, :num_ao] = kin[:num_ao, num_ao:].conj().T

        return kin

    def _int_pot() -> np.ndarray:
        def _pw_pw_pot() -> np.ndarray:
            delta_k, delta_k2 = _pw_pair_transfer(pw_k)
            pw_pw = np.zeros(delta_k2.shape, dtype=complex)
            nonzero = delta_k2 > 1e-14

            for nuc_id in range(mol.natm):
                charge = float(mol.atom_charge(nuc_id))
                center = np.asarray(mol.atom_coord(nuc_id), dtype=float)
                phase = np.exp(1j * np.einsum("pqi,i->pq", delta_k, center))
                pw_pw[nonzero] -= 4.0 * pi * charge * phase[nonzero] / delta_k2[nonzero]

            return pw_norm * pw_norm * pw_pw

        pot = np.zeros((num_basis, num_basis), dtype=complex)

        pot[:num_ao, :num_ao] = mol.intor('int1e_nuc_sph', hermi=1)

        pot[num_ao:, num_ao:] = _pw_pw_pot()
        pot[:num_ao, num_ao:] = ao_pw_pot(mol, pw_k, pw_norm)

        pot[num_ao:, :num_ao] = pot[:num_ao, num_ao:].conj().T

        return pot

    def _int_dip() -> np.ndarray:
        dip = np.zeros((3, num_basis, num_basis), dtype=complex)

        dip[:, :num_ao, :num_ao] = mol.intor('int1e_r_sph', comp=3)

        # dip[]:, num_ao:, num_ao:] = 0.0

        ao_pw = ao_pw_dipole(mol, pw_k, pw_norm)
        dip[:, :num_ao, num_ao:] = ao_pw
        dip[:, num_ao:, :num_ao] = dip[:, :num_ao, num_ao:].conj().transpose(0, 2, 1)

        return dip

    # overlap term
    if op.lower() == "i":
        return _int_ovlp()
    # kinetic energy term
    elif op.lower() == "kin":
        return _int_kin()
    # potential energy term
    elif op.lower() == "pot":
        return _int_pot()
    # dipole operator term
    elif op.lower() == "r":
        return _int_dip()
    else:
        raise ValueError(f"Unknown operator: {op}")


def int_2e(mol: gto.Mole, pw_k: np.ndarray, int2e_types: tuple[str, ...],
           pw_norm: float = 1.0, cutoff: float = -1.0) -> list[INT2e]:
    '''
    Calculate the two-electron integral of the plane wave with the Gaussian basis functions
    Args:
        mol: PySCF molecule object.
        pw_k: Plane wave vectors.
        int2e_types: Integral types to generate.
        pw_norm: Normalization factor for the plane waves.
        cutoff: Threshold for ignoring small integrals.
    Returns:
        Two-electron integral matrix.
    '''
    
    def _pair_idx(idx_pair: int) -> tuple[int, int, int, int]:
        def _unpack_tril(idx: int) -> tuple[int, int]:
            first = int((np.sqrt(8 * idx + 1) - 1) // 2)
            second = idx - first * (first + 1) // 2
            return first, second

        ij, kl = _unpack_tril(idx_pair)
        i, j = _unpack_tril(ij)
        k, l = _unpack_tril(kl)

        return i, j, k, l

    def _shell_cart2sph_matrix(angular: int, num_contract: int) -> np.ndarray:
        cart2sph = np.asarray(gto.cart2sph(angular), dtype=float)
        num_cart, num_sph = cart2sph.shape
        transform = np.zeros((num_contract * num_sph, num_cart * num_contract), dtype=float)
        for contract_id in range(num_contract):
            for cart_id in range(num_cart):
                start = contract_id * num_sph
                stop = start + num_sph
                transform[start:stop, cart_id * num_contract + contract_id] = cart2sph[cart_id]

        return transform

    def _build_shells() -> list[tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int, int]],
                                      np.ndarray, int, int]]:
        ao_loc = mol.ao_loc_nr()
        shells = []
        for shell_id in range(mol.nbas):
            angular = mol.bas_angular(shell_id)
            exponents = np.asarray(mol.bas_exp(shell_id), dtype=float)
            coefficients = np.asarray(mol.bas_ctr_coeff(shell_id), dtype=float)
            coefficients *= gto.gto_norm(angular, exponents)[:, None]
            shells.append((
                np.asarray(mol.atom_coord(mol.bas_atom(shell_id)), dtype=float),
                exponents,
                coefficients,
                _cartesian_powers(angular),
                _shell_cart2sph_matrix(angular, coefficients.shape[1]),
                int(ao_loc[shell_id]),
                int(ao_loc[shell_id + 1]),
            ))

        return shells

    def _cart2sph_1ao(transform_a: np.ndarray, shell_cart: np.ndarray) -> np.ndarray:
        return np.einsum("ap,nmrp->nmra", transform_a, shell_cart, optimize=True)

    def _cart2sph_2ao(transform_a: np.ndarray, transform_b: np.ndarray,
        shell_cart: np.ndarray) -> np.ndarray:
        return np.einsum("ap,bq,nmpq->nmab", transform_a, transform_b, shell_cart,
                         optimize=True)

    def _cart2sph_3ao(transform_a: np.ndarray, transform_b: np.ndarray,
        transform_c: np.ndarray, shell_cart: np.ndarray) -> np.ndarray:
        return np.einsum("ap,bq,cr,npqr->nabc", transform_a, transform_b, transform_c,
                         shell_cart, optimize=True)

    def _fourier_moment(power: int, exponent: float, k: np.ndarray) -> np.ndarray:
        moments = [sqrt(pi / exponent) * np.exp(-(k * k) / (4.0 * exponent)) + 0.j]
        if power == 0:
            return moments[0]

        moments.append(1j * k * moments[0] / (2.0 * exponent))
        for order in range(1, power):
            moments.append((order * moments[order - 1] + 1j * k * moments[order]) / (2.0 * exponent))

        return moments[power]

    def _pair_hermite_coeff(power_a: int, power_b: int, shift_a: float, shift_b: float,
        exponent: float) -> np.ndarray:
        coeffs = np.zeros(power_a + power_b + 1, dtype=complex)
        for centered_a in range(power_a + 1):
            coeff_a = comb(power_a, centered_a) * shift_a ** (power_a - centered_a)
            for centered_b in range(power_b + 1):
                coeff_b = comb(power_b, centered_b) * shift_b ** (power_b - centered_b)
                hermite_coeffs = _central_hermite_coeff(centered_a + centered_b, exponent)
                for order, coeff in enumerate(hermite_coeffs):
                    coeffs[order] += coeff_a * coeff_b * coeff

        return coeffs

    def _plain_pair_coeff(power_a: int, power_b: int, shift_a: float, shift_b: float) -> np.ndarray:
        coeffs = np.zeros(power_a + power_b + 1, dtype=complex)
        for centered_a in range(power_a + 1):
            coeff_a = comb(power_a, centered_a) * shift_a ** (power_a - centered_a)
            for centered_b in range(power_b + 1):
                coeff_b = comb(power_b, centered_b) * shift_b ** (power_b - centered_b)
                coeffs[centered_a + centered_b] += coeff_a * coeff_b

        return coeffs

    def _pw_shifted_center_phase(center: np.ndarray, exponent: float, kvec: np.ndarray,
        sign: float) -> tuple[np.ndarray, np.ndarray]:
        k2 = np.einsum("...i,...i->...", kvec, kvec)
        complex_center = center + 1j * sign * kvec / (2.0 * exponent)
        phase = np.exp(1j * sign * np.einsum("...i,i->...", kvec, center) - k2 / (4.0 * exponent))

        return complex_center, phase

    def _shifted_hermite_coeff_xyz(powers: tuple[int, int, int], complex_center: np.ndarray,
        real_center: np.ndarray, exponent: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            _shifted_hermite_coeff(powers[0], complex_center[:, 0] - real_center[0], exponent),
            _shifted_hermite_coeff(powers[1], complex_center[:, 1] - real_center[1], exponent),
            _shifted_hermite_coeff(powers[2], complex_center[:, 2] - real_center[2], exponent),
        )

    def _contract_hermite_coulomb(coeff_a: tuple[np.ndarray, np.ndarray, np.ndarray],
        coeff_b: tuple[np.ndarray, np.ndarray, np.ndarray], rho: float, pc: np.ndarray,
        boys_arg: np.ndarray) -> np.ndarray:
        coeff_ax, coeff_ay, coeff_az = coeff_a
        coeff_bx, coeff_by, coeff_bz = coeff_b
        value = np.zeros(pc.shape[0], dtype=complex)
        for tx_a, cx_a in enumerate(coeff_ax):
            for ty_a, cy_a in enumerate(coeff_ay):
                coeff_xy_a = cx_a * cy_a
                for tz_a, cz_a in enumerate(coeff_az):
                    coeff_a_value = coeff_xy_a * cz_a
                    for tx_b, cx_b in enumerate(coeff_bx):
                        for ty_b, cy_b in enumerate(coeff_by):
                            coeff_xy_b = cx_b * cy_b
                            for tz_b, cz_b in enumerate(coeff_bz):
                                sign = -1.0 if (tx_b + ty_b + tz_b) % 2 else 1.0
                                coeff_b_value = sign * coeff_xy_b * cz_b
                                value += coeff_a_value * coeff_b_value * _hermite_coulomb(
                                    tx_a + tx_b, ty_a + ty_b, tz_a + tz_b, 0, rho, pc, boys_arg)

        return value

    def _mixed_pw_ao_coulomb(alpha: float, center_q: np.ndarray, powers_q: tuple[int, int, int],
        beta: float, center_s: np.ndarray, powers_s: tuple[int, int, int], sign_s: float) -> np.ndarray:
        rho = alpha * beta / (alpha + beta)

        center_pq, phase_q = _pw_shifted_center_phase(center_q, alpha, pw_k, -1.0)
        center_rs, phase_s = _pw_shifted_center_phase(center_s, beta, pw_k, sign_s)
        center_pq = center_pq[:, None, :]
        center_rs = center_rs[None, :, :]
        center_pq = np.broadcast_to(center_pq, (num_pw, num_pw, 3)).reshape(-1, 3)
        center_rs = np.broadcast_to(center_rs, (num_pw, num_pw, 3)).reshape(-1, 3)

        pc = center_pq - center_rs
        boys_arg = rho * np.einsum("pi,pi->p", pc, pc)
        prefactor = (pw_norm * pw_norm * phase_q[:, None] * phase_s[None, :]
                     * 2.0 * pi ** 2.5 / (alpha * beta * sqrt(alpha + beta)))

        coeff_q = _shifted_hermite_coeff_xyz(powers_q, center_pq, center_q, alpha)
        coeff_s = _shifted_hermite_coeff_xyz(powers_s, center_rs, center_s, beta)
        value = _contract_hermite_coulomb(coeff_q, coeff_s, rho, pc, boys_arg)

        return prefactor * value.reshape(num_pw, num_pw)

    def _primitive_bbbb() -> np.ndarray:
        return mol.intor('int2e_sph', aosym="s8")

    def _primitive_kbbb(alpha: float, center_j: np.ndarray, powers_j: tuple[int, int, int],
        beta: float, center_k: np.ndarray, powers_k: tuple[int, int, int], gamma: float,
        center_l: np.ndarray, powers_l: tuple[int, int, int]) -> np.ndarray:
        eta = beta + gamma
        rho = alpha * eta / (alpha + eta)
        center_kl = (beta * center_k + gamma * center_l) / eta
        pair_prefactor = np.exp(-beta * gamma / eta * np.dot(center_k - center_l, center_k - center_l))

        center_pw, phase = _pw_shifted_center_phase(center_j, alpha, pw_k, -1.0)
        pc = center_pw - center_kl[None, :]
        boys_arg = rho * np.einsum("pi,pi->p", pc, pc)
        prefactor = pw_norm * phase * pair_prefactor * 2.0 * pi ** 2.5 / (alpha * eta * sqrt(alpha + eta))

        coeff_j = _shifted_hermite_coeff_xyz(powers_j, center_pw, center_j, alpha)
        coeff_klx = _pair_hermite_coeff(powers_k[0], powers_l[0], center_kl[0] - center_k[0],
                                        center_kl[0] - center_l[0], eta)
        coeff_kly = _pair_hermite_coeff(powers_k[1], powers_l[1], center_kl[1] - center_k[1],
                                        center_kl[1] - center_l[1], eta)
        coeff_klz = _pair_hermite_coeff(powers_k[2], powers_l[2], center_kl[2] - center_k[2],
                                        center_kl[2] - center_l[2], eta)
        coeff_kl = (coeff_klx, coeff_kly, coeff_klz)
        value = _contract_hermite_coulomb(coeff_j, coeff_kl, rho, pc, boys_arg)

        return prefactor * value

    def _primitive_kkbb(alpha: float, center_r: np.ndarray, powers_r: tuple[int, int, int],
        beta: float, center_s: np.ndarray, powers_s: tuple[int, int, int]) -> np.ndarray:
        eta = alpha + beta
        center_rs = (alpha * center_r + beta * center_s) / eta
        pair_prefactor = np.exp(-alpha * beta / eta * np.dot(center_r - center_s, center_r - center_s))

        delta_k, delta_k2 = _pw_pair_transfer(pw_k)
        flat_delta_k = delta_k.reshape(-1, 3)
        flat_delta_k2 = delta_k2.reshape(-1)
        nonzero = flat_delta_k2 > 1e-14

        value = np.zeros(num_pw * num_pw, dtype=complex)
        if not np.any(nonzero):
            return value.reshape(num_pw, num_pw)

        active_k = flat_delta_k[nonzero]
        phase = np.exp(1j * np.einsum("pi,i->p", active_k, center_rs))
        coeff_x = _plain_pair_coeff(powers_r[0], powers_s[0], center_rs[0] - center_r[0],
                                    center_rs[0] - center_s[0])
        coeff_y = _plain_pair_coeff(powers_r[1], powers_s[1], center_rs[1] - center_r[1],
                                    center_rs[1] - center_s[1])
        coeff_z = _plain_pair_coeff(powers_r[2], powers_s[2], center_rs[2] - center_r[2],
                                    center_rs[2] - center_s[2])

        moment = np.zeros(active_k.shape[0], dtype=complex)
        for tx, cx in enumerate(coeff_x):
            for ty, cy in enumerate(coeff_y):
                coeff_xy = cx * cy
                for tz, cz in enumerate(coeff_z):
                    moment += (coeff_xy * cz * _fourier_moment(tx, eta, active_k[:, 0])
                               * _fourier_moment(ty, eta, active_k[:, 1])
                               * _fourier_moment(tz, eta, active_k[:, 2]))

        value[nonzero] = (pw_norm * pw_norm * 4.0 * pi / flat_delta_k2[nonzero]
                          * pair_prefactor * phase * moment)

        return value.reshape(num_pw, num_pw)

    def _primitive_kbkb(alpha: float, center_q: np.ndarray, powers_q: tuple[int, int, int],
        beta: float, center_s: np.ndarray, powers_s: tuple[int, int, int]) -> np.ndarray:
        return _mixed_pw_ao_coulomb(alpha, center_q, powers_q, beta, center_s, powers_s, -1.0)

    def _primitive_kbbk(alpha: float, center_q: np.ndarray, powers_q: tuple[int, int, int],
        beta: float, center_s: np.ndarray, powers_s: tuple[int, int, int]) -> np.ndarray:
        return _mixed_pw_ao_coulomb(alpha, center_q, powers_q, beta, center_s, powers_s, 1.0)

    def _primitive_kkkb(alpha: float, center_a: np.ndarray, powers_a: tuple[int, int, int]) -> np.ndarray:
        delta_k, delta_k2 = _pw_pair_transfer(pw_k)
        momentum = delta_k[:, :, None, :] - pw_k[None, None, :, :]
        flat_momentum = momentum.reshape(-1, 3)
        flat_delta_k2 = np.broadcast_to(delta_k2[:, :, None], momentum.shape[:-1]).reshape(-1)
        nonzero = flat_delta_k2 > 1e-14

        value = np.zeros(num_pw * num_pw * num_pw, dtype=complex)
        if not np.any(nonzero):
            return value.reshape(num_pw, num_pw, num_pw)

        active_k = flat_momentum[nonzero]
        phase = np.exp(1j * np.einsum("pi,i->p", active_k, center_a))
        moment = (phase * _fourier_moment(powers_a[0], alpha, active_k[:, 0])
                  * _fourier_moment(powers_a[1], alpha, active_k[:, 1])
                  * _fourier_moment(powers_a[2], alpha, active_k[:, 2]))
        value[nonzero] = pw_norm ** 3 * 4.0 * pi * moment / flat_delta_k2[nonzero]

        return value.reshape(num_pw, num_pw, num_pw)

    def _primitive_kkkk() -> tuple[np.ndarray, np.ndarray]:
        def _is_canonical(p: int, q: int, r: int, s: int) -> bool:
            index = (p, q, r, s)
            return index == max(index, (r, s, p, q), (q, p, s, r), (s, r, q, p))

        delta_k, delta_k2_pair = _pw_pair_transfer(pw_k)
        delta_k_left = delta_k[:, :, None, None, :]
        delta_k_right = pw_k[None, None, None, :, :] - pw_k[None, None, :, None, :]
        total_k = delta_k_left + delta_k_right
        total_k2 = np.einsum("pqxyi,pqxyi->pqxy", total_k, total_k)
        delta_k2 = np.broadcast_to(delta_k2_pair[:, :, None, None], total_k2.shape)
        nonzero = (delta_k2 > 1e-14) & (total_k2 < 1e-14)
        canonical = np.zeros(nonzero.shape, dtype=bool)
        for p in range(num_pw):
            for q in range(num_pw):
                for r in range(num_pw):
                    for s in range(num_pw):
                        canonical[p, q, r, s] = _is_canonical(p, q, r, s)

        value = np.zeros((num_pw, num_pw, num_pw, num_pw), dtype=complex)
        value[nonzero] = pw_norm * pw_norm * 4.0 * pi / delta_k2[nonzero]
        print('Check again why there are only two pw_norm being multiplied above instead of four.')
        quit()

        return value, canonical

    int2e = []
    num_ao = mol.nao_nr()
    num_pw, _ = pw_k.shape

    # 0pw4ao terms (b, b, b, b)
    if 'bbbb' in int2e_types:
        int2e_ao = _primitive_bbbb()
        num_pairs = num_ao * (num_ao + 1) // 2
        for n in range(num_pairs * (num_pairs + 1) // 2):
            value = int2e_ao[n] + 0.j
            if abs(value) > cutoff:
                i, j, k, l = _pair_idx(n)
                int2e.append(INT2e((i, j, k, l), value, 'bbbb'))

    shells = _build_shells()

    # 1pw3ao terms (k, b, b, b)
    kbbb_shells = shells if 'kbbb' in int2e_types else []
    for data_j in kbbb_shells:
        center_j, exponents_j, coefficients_j, powers_j, transform_j, start_j, stop_j = data_j
        for shell_k, data_k in enumerate(shells):
            center_k, exponents_k, coefficients_k, powers_k, transform_k, start_k, stop_k = data_k
            for shell_l in range(shell_k + 1):
                center_l, exponents_l, coefficients_l, powers_l, transform_l, start_l, stop_l = shells[shell_l]
                shell_cart = np.zeros((num_pw, len(powers_j), coefficients_j.shape[1],
                                       len(powers_k), coefficients_k.shape[1],
                                       len(powers_l), coefficients_l.shape[1]), dtype=complex)

                for prim_j, alpha in enumerate(exponents_j):
                    for prim_k, beta in enumerate(exponents_k):
                        for prim_l, gamma in enumerate(exponents_l):
                            contract_coeff = np.einsum("i,j,k->ijk", coefficients_j[prim_j],
                                                       coefficients_k[prim_k], coefficients_l[prim_l])
                            for cart_j, power_j in enumerate(powers_j):
                                for cart_k, power_k in enumerate(powers_k):
                                    for cart_l, power_l in enumerate(powers_l):
                                        primitive = _primitive_kbbb(alpha, center_j, power_j, beta,
                                                                   center_k, power_k, gamma,
                                                                   center_l, power_l)
                                        shell_cart[:, cart_j, :, cart_k, :, cart_l, :] += (
                                            primitive[:, None, None, None] * contract_coeff[None, :, :, :])

                shell_cart = shell_cart.reshape(num_pw, len(powers_j) * coefficients_j.shape[1],
                                                len(powers_k) * coefficients_k.shape[1],
                                                len(powers_l) * coefficients_l.shape[1])
                shell_ao = _cart2sph_3ao(transform_j, transform_k, transform_l, shell_cart)

                for pw_id in range(num_pw):
                    p = num_ao + pw_id
                    for local_j, q in enumerate(range(start_j, stop_j)):
                        for local_k, r in enumerate(range(start_k, stop_k)):
                            for local_l, s in enumerate(range(start_l, stop_l)):
                                if r < s:
                                    continue
                                value = shell_ao[pw_id, local_j, local_k, local_l]
                                if abs(value) > cutoff:
                                    int2e.append(INT2e((p, q, r, s), value, 'kbbb'))

    # 2pw2ao terms (k, b, b, k) and (k, b, k, b)
    mixed_pair_shells = shells if {'kbbk', 'kbkb'} & int2e_types else []
    for data_q in mixed_pair_shells:
        center_q, exponents_q, coefficients_q, powers_q, transform_q, start_q, stop_q = data_q
        for data_s in shells:
            center_s, exponents_s, coefficients_s, powers_s, transform_s, start_s, stop_s = data_s
            shell_cart_kbbk = np.zeros((num_pw, num_pw, len(powers_q), coefficients_q.shape[1],
                                        len(powers_s), coefficients_s.shape[1]), dtype=complex)
            shell_cart_kbkb = np.zeros((num_pw, num_pw, len(powers_q), coefficients_q.shape[1],
                                        len(powers_s), coefficients_s.shape[1]), dtype=complex)

            for prim_q, alpha in enumerate(exponents_q):
                for prim_s, beta in enumerate(exponents_s):
                    contract_coeff = np.einsum("i,j->ij", coefficients_q[prim_q], coefficients_s[prim_s])
                    for cart_q, power_q in enumerate(powers_q):
                        for cart_s, power_s in enumerate(powers_s):
                            primitive_kbbk = _primitive_kbbk(alpha, center_q, power_q, beta,
                                                            center_s, power_s)
                            primitive_kbkb = _primitive_kbkb(alpha, center_q, power_q, beta,
                                                            center_s, power_s)
                            shell_cart_kbbk[:, :, cart_q, :, cart_s, :] += (
                                primitive_kbbk[:, :, None, None] * contract_coeff[None, None, :, :])
                            shell_cart_kbkb[:, :, cart_q, :, cart_s, :] += (
                                primitive_kbkb[:, :, None, None] * contract_coeff[None, None, :, :])

            shell_cart_kbbk = shell_cart_kbbk.reshape(num_pw, num_pw,
                                                      len(powers_q) * coefficients_q.shape[1],
                                                      len(powers_s) * coefficients_s.shape[1])
            shell_cart_kbkb = shell_cart_kbkb.reshape(num_pw, num_pw,
                                                      len(powers_q) * coefficients_q.shape[1],
                                                      len(powers_s) * coefficients_s.shape[1])
            shell_ao_kbbk = _cart2sph_2ao(transform_q, transform_s, shell_cart_kbbk)
            shell_ao_kbkb = _cart2sph_2ao(transform_q, transform_s, shell_cart_kbkb)

            for pw_id_p in range(num_pw):
                p = num_ao + pw_id_p
                for local_q, q in enumerate(range(start_q, stop_q)):
                    for pw_id_r in range(num_pw):
                        if pw_id_p < pw_id_r:
                            continue
                        r = num_ao + pw_id_r
                        for local_s, s in enumerate(range(start_s, stop_s)):
                            if pw_id_p == pw_id_r and q < s:
                                continue
                            if 'kbbk' in int2e_types:
                                value = shell_ao_kbbk[pw_id_p, pw_id_r, local_q, local_s]
                                if abs(value) > cutoff:
                                    int2e.append(INT2e((p, q, s, r), value, 'kbbk'))
                            if 'kbkb' in int2e_types:
                                value = shell_ao_kbkb[pw_id_p, pw_id_r, local_q, local_s]
                                if abs(value) > cutoff:
                                    int2e.append(INT2e((p, q, r, s), value, 'kbkb'))

    # 2pw2ao terms (k, k, b, b)
    if 'kkbb' in int2e_types:
        for shell_r, data_r in enumerate(shells):
            center_r, exponents_r, coefficients_r, powers_r, transform_r, start_r, stop_r = data_r
            for shell_s in range(shell_r + 1):
                center_s, exponents_s, coefficients_s, powers_s, transform_s, start_s, stop_s = shells[shell_s]
                shell_cart = np.zeros((num_pw, num_pw, len(powers_r), coefficients_r.shape[1],
                                       len(powers_s), coefficients_s.shape[1]), dtype=complex)

                for prim_r, alpha in enumerate(exponents_r):
                    for prim_s, beta in enumerate(exponents_s):
                        contract_coeff = np.einsum("i,j->ij", coefficients_r[prim_r], coefficients_s[prim_s])
                        for cart_r, power_r in enumerate(powers_r):
                            for cart_s, power_s in enumerate(powers_s):
                                primitive = _primitive_kkbb(alpha, center_r, power_r, beta, center_s, power_s)
                                shell_cart[:, :, cart_r, :, cart_s, :] += (
                                    primitive[:, :, None, None] * contract_coeff[None, None, :, :])

                shell_cart = shell_cart.reshape(num_pw, num_pw, len(powers_r) * coefficients_r.shape[1],
                                                len(powers_s) * coefficients_s.shape[1])
                shell_ao = _cart2sph_2ao(transform_r, transform_s, shell_cart)

                for pw_id_p in range(num_pw):
                    p = num_ao + pw_id_p
                    for pw_id_q in range(pw_id_p + 1):
                        q = num_ao + pw_id_q
                        for local_r, r in enumerate(range(start_r, stop_r)):
                            for local_s, s in enumerate(range(start_s, stop_s)):
                                if r < s:
                                    continue
                                value = shell_ao[pw_id_p, pw_id_q, local_r, local_s]
                                if abs(value) > cutoff:
                                    int2e.append(INT2e((p, q, r, s), value, 'kkbb'))

    # 3pw1ao terms (k, k, k, b)
    if 'kkkb' in int2e_types:
        for data_a in shells:
            center_a, exponents_a, coefficients_a, powers_a, transform_a, start_a, stop_a = data_a
            shell_cart = np.zeros((num_pw, num_pw, num_pw, len(powers_a), coefficients_a.shape[1]), dtype=complex)

            for prim_a, alpha in enumerate(exponents_a):
                for cart_a, power_a in enumerate(powers_a):
                    primitive = _primitive_kkkb(alpha, center_a, power_a)
                    shell_cart[:, :, :, cart_a, :] += primitive[:, :, :, None] * coefficients_a[prim_a, None, :]

            shell_cart = shell_cart.reshape(num_pw, num_pw, num_pw, len(powers_a) * coefficients_a.shape[1])
            shell_ao = _cart2sph_1ao(transform_a, shell_cart)

            for pw_id_p in range(num_pw):
                p = num_ao + pw_id_p
                for pw_id_q in range(num_pw):
                    q = num_ao + pw_id_q
                    for pw_id_r in range(num_pw):
                        r = num_ao + pw_id_r
                        for local_a, a in enumerate(range(start_a, stop_a)):
                            value = shell_ao[pw_id_p, pw_id_q, pw_id_r, local_a]
                            if abs(value) > cutoff:
                                int2e.append(INT2e((p, q, r, a), value, 'kkkb'))

    # 4pw terms (k, k, k, k)
    if 'kkkk' in int2e_types:
        int2e_kkkk, canonical_kkkk = _primitive_kkkk()
        for pw_id_p in range(num_pw):
            p = num_ao + pw_id_p
            for pw_id_q in range(num_pw):
                q = num_ao + pw_id_q
                for pw_id_r in range(num_pw):
                    r = num_ao + pw_id_r
                    for pw_id_s in range(num_pw):
                        s = num_ao + pw_id_s
                        if not canonical_kkkk[pw_id_p, pw_id_q, pw_id_r, pw_id_s]:
                            continue
                        value = int2e_kkkk[pw_id_p, pw_id_q, pw_id_r, pw_id_s]
                        if abs(value) > cutoff:
                            int2e.append(INT2e((p, q, r, s), value, 'kkkk'))

    return int2e


def _expand_int2e_mopw(int2e_mopw: list[INT2e], basis_num: int) -> np.ndarray:

    '''
    Imam
    Constructs the full 2e integral, which includes matrix elements related by symmetries.
    '''
    
    int2e_full = np.zeros((basis_num, basis_num, basis_num, basis_num), dtype=complex)

    def set_int2e(i: int, j: int, k: int, l: int, value: complex) -> None:
        if any(index < 0 or index >= basis_num for index in (i, j, k, l)):
            raise ValueError(f"Invalid transformed integral index: {(i, j, k, l)}")
        int2e_full[i, j, k, l] = value

    for integral in int2e_mopw:
        i, j, k, l = integral.index
        value = integral.value

        if integral.type == 'bbbb':
            for a, b in ((i, j), (j, i)):
                for c, d in ((k, l), (l, k)):
                    set_int2e(a, b, c, d, value)
                    set_int2e(c, d, a, b, value)
        elif integral.type == 'kbbb':
            for b, c in ((k, l), (l, k)):
                set_int2e(i, j, b, c, value)
                set_int2e(b, c, i, j, value)
                set_int2e(j, i, c, b, np.conj(value))
                set_int2e(c, b, j, i, np.conj(value))
        elif integral.type == 'kkbb':
            for a, b in ((k, l), (l, k)):
                set_int2e(i, j, a, b, value)
                set_int2e(a, b, i, j, value)
                set_int2e(j, i, b, a, np.conj(value))
                set_int2e(b, a, j, i, np.conj(value))
        elif integral.type in {'kbbk', 'kbkb'}:
            set_int2e(i, j, k, l, value)
            set_int2e(k, l, i, j, value)
            set_int2e(j, i, l, k, np.conj(value))
            set_int2e(l, k, j, i, np.conj(value))
        elif integral.type == 'kkkb':
            set_int2e(i, j, k, l, value)
            set_int2e(k, l, i, j, value)
            set_int2e(j, i, l, k, np.conj(value))
            set_int2e(l, k, j, i, np.conj(value))
        elif integral.type == 'kkkk':
            set_int2e(i, j, k, l, value)
            set_int2e(k, l, i, j, value)
            set_int2e(j, i, l, k, np.conj(value))
            set_int2e(l, k, j, i, np.conj(value))
        else:
            raise ValueError(f"Unknown integral type: {integral.type}")

    return int2e_full


def _drop_inactive_int2e(int2e_mopw: list[INT2e], inactive_num: int, active_num: int,
    pw_num: int) -> list[INT2e]:
    full_bound_num = inactive_num + active_num

    def map_index(index: int) -> int | None:
        if index < inactive_num:
            return None
        if index < full_bound_num:
            return index - inactive_num
        if index < full_bound_num + pw_num:
            return active_num + index - full_bound_num
        raise ValueError(f"Invalid transformed integral index: {index}")

    retained_int2e = []
    for integral in int2e_mopw:
        mapped = [map_index(index) for index in integral.index]
        if any(index is None for index in mapped):
            continue
        retained_int2e.append(INT2e(tuple(index for index in mapped if index is not None), # type: ignore
                                    integral.value, integral.type))

    return retained_int2e


def core_energy_ao2mo(imo_coeff: np.ndarray, amo_coeff: np.ndarray, int1e_ao: np.ndarray,
    int2e_mopw: list[INT2e], ao_num: int, pw_num: int,
    nuclear_repulsion: float = 0.0) -> complex:
    imo_ao_num, imo_num = imo_coeff.shape
    amo_ao_num, amo_num = amo_coeff.shape
    assert imo_ao_num == ao_num, "Inactive MO coefficients shape does not match the number of AOs."
    assert amo_ao_num == ao_num, "Active MO coefficients shape does not match the number of AOs."

    if imo_num == 0:
        return 0.0 + 0.0j

    full_mopw_num = imo_num + amo_num + pw_num
    int1e_core = imo_coeff.conj().T @ int1e_ao[:ao_num, :ao_num] @ imo_coeff
    int2e_full = _expand_int2e_mopw(int2e_mopw, full_mopw_num)

    core_energy = nuclear_repulsion + 2.0 * np.trace(int1e_core)
    for i in range(imo_num):
        for j in range(imo_num):
            core_energy += 2.0 * int2e_full[i, i, j, j]
            core_energy -= int2e_full[i, j, j, i]

    return core_energy


def int1e_ao2mo(amo_coeff: np.ndarray, int1e_ao: np.ndarray, ao_num: int, pw_num: int, 
                imo_coeff: np.ndarray = None, int2e_mopw: list[INT2e] = None) -> np.ndarray:
    '''
    Transform one-electron integrals from AO+PW basis to inactive-MO+active-MO+PW basis,
    add the frozen-core mean-field contribution, then return the active-MO+PW block.
    Args:
        amo_coeff: AO-to-active-MO coefficient matrix with shape (num_ao, num_active_mo).
        int1e_ao: One-electron integral matrix in AO+PW basis.
        
        num_ao: Number of atomic orbitals.
        num_pw: Number of plane waves.
        imo_coeff: AO-to-inactive-MO coefficient matrix with shape (num_ao, num_inactive_mo).
        int2e_mopw: Two-electron integral list in inactive-MO+active-MO+PW basis. Must be 
           provided when imo_coeff is provided.
    Returns:
        Effective one-electron integral matrix in active-MO+PW basis.
    '''

    if imo_coeff is not None:
        imo_ao_num, imo_num = imo_coeff.shape
        assert imo_ao_num == ao_num, "Inactive MO coefficients shape does not match the number of AOs."
        assert int2e_mopw is not None, "int2e_mopw is needed when imo_coeff is given."
    else:
        imo_num = 0
    
    amo_ao_num, amo_num = amo_coeff.shape
    aopw_num0,_ = int1e_ao.shape
    assert amo_ao_num == ao_num, "Active MO coefficients shape does not match the number of AOs."
    assert aopw_num0 == ao_num + pw_num, "One-electron integral shape does not match the AO+PW basis."
    aopw_num = ao_num + pw_num
    full_mopw_num = imo_num + amo_num + pw_num

    transform = np.zeros((aopw_num, full_mopw_num), dtype=complex)
    if imo_coeff is not None:
        transform[:ao_num, :imo_num] = imo_coeff
    transform[:ao_num, imo_num:imo_num + amo_num] = amo_coeff
    transform[ao_num:, imo_num + amo_num:] = np.eye(pw_num, dtype=complex)

    int1e_full = transform.conj().T @ int1e_ao @ transform

    if imo_coeff is None:
        return int1e_full
    else:
        int2e_full = _expand_int2e_mopw(int2e_mopw, full_mopw_num)    # imam: is there a more memory-friendly way of doing this?
        for core_id in range(imo_num):
            int1e_full += 2.0 * int2e_full[:, :, core_id, core_id]      # imam: add Coulomb interaction with core.
            int1e_full -= int2e_full[:, core_id, core_id, :]            # imam: add exchange interaction with core.
    
        return int1e_full[imo_num:, imo_num:]


#imam - old def int2e_ao2mo(mo_coeff: np.ndarray, int2e_ao: list[INT2e], ao_num: int, pw_num: int,
#imam - old     int2e_types: tuple[str, ...], cutoff: float = -1.0) -> list[INT2e]:
def int2e_ao2mo(mo_coeff: np.ndarray, int2e_ao: list[INT2e], ao_num: int, pw_num: int,
    cutoff: float = -1.0) -> list[INT2e]:
    '''
    Transform the two-electron integrals from AO+PW basis to MO+PW basis.
    Args:
        mo_coeff: AO-to-MO coefficient matrix with shape (num_ao, num_mo).
        int2e_ao: Two-electron integral list in AO+PW basis.
        num_ao: Number of atomic orbitals.
        num_pw: Number of plane waves.
    Returns:
        Two-electron integral list in MO+PW basis.

    Imam - Removed:
        int2e_types: Integral classes to transform. Use None for all supported classes.
    '''
    ao_num0, mo_num = mo_coeff.shape
    assert ao_num0 == ao_num, "MO coefficients shape does not match the number of AOs."

    def _pw_id(index: int, integral: INT2e) -> int:
        pw_id = index - ao_num
        if pw_id < 0 or pw_id >= pw_num:
            raise ValueError(f"Invalid {integral.type} integral index: {integral.index}")

        return pw_id

    def _check_ao_indices(integral: INT2e, *indices: int) -> None:
        if any(index < 0 or index >= ao_num for index in indices):
            raise ValueError(f"Invalid {integral.type} integral index: {integral.index}")

    def _is_kkkk_canonical(p: int, q: int, r: int, s: int) -> bool:
        '''
        imam
        Returns true only for independent elements. Independent elements have p>=q, r>=s, and
        pq >= rs.
        '''
        index = (p, q, r, s)
        return index == max(index, (r, s, p, q), (q, p, s, r), (s, r, q, p))

    has_bbbb = False
    int2e_bbbb = np.zeros((ao_num, ao_num, ao_num, ao_num), dtype=complex)
    int2e_kbbb = np.zeros((pw_num, ao_num, ao_num, ao_num), dtype=complex)
    has_kbbb = False
    int2e_kkbb = np.zeros((pw_num, pw_num, ao_num, ao_num), dtype=complex)
    has_kkbb = False
    int2e_kbbk = np.zeros((pw_num, ao_num, ao_num, pw_num), dtype=complex)
    has_kbbk = False
    int2e_kbkb = np.zeros((pw_num, ao_num, pw_num, ao_num), dtype=complex)
    has_kbkb = False
    int2e_kkkb = np.zeros((pw_num, pw_num, pw_num, ao_num), dtype=complex)
    has_kkkb = False
    int2e_kkkk = np.zeros((pw_num, pw_num, pw_num, pw_num), dtype=complex)
    has_kkkk = False
    for integral in int2e_ao:
        #print('imam - check: has ALL_INT2E_TYPES been defined?')
        if integral.type not in set(ALL_INT2E_TYPES):
            raise ValueError(f"Unknown integral type: {integral.type}")
        #imam: why? if integral.type not in enabled_types:
        #imam: why?     continue

        if integral.type == 'bbbb':
            i, j, k, l = integral.index
            value = integral.value
            _check_ao_indices(integral, i, j, k, l)
            int2e_bbbb[i, j, k, l] = value
            int2e_bbbb[j, i, k, l] = value
            int2e_bbbb[i, j, l, k] = value
            int2e_bbbb[j, i, l, k] = value
            int2e_bbbb[k, l, i, j] = value
            int2e_bbbb[l, k, i, j] = value
            int2e_bbbb[k, l, j, i] = value
            int2e_bbbb[l, k, j, i] = value
            has_bbbb = True
        elif integral.type == 'kbbb':
            p, q, r, s = integral.index
            pw_id = _pw_id(p, integral)
            _check_ao_indices(integral, q, r, s)
            value = integral.value
            int2e_kbbb[pw_id, q, r, s] = value
            int2e_kbbb[pw_id, q, s, r] = value
            has_kbbb = True
        elif integral.type == 'kkbb':
            p, q, r, s = integral.index
            pw_id_p = _pw_id(p, integral)
            pw_id_q = _pw_id(q, integral)
            _check_ao_indices(integral, r, s)
            value = integral.value
            int2e_kkbb[pw_id_p, pw_id_q, r, s] = value
            int2e_kkbb[pw_id_p, pw_id_q, s, r] = value
            int2e_kkbb[pw_id_q, pw_id_p, s, r] = np.conj(value)
            int2e_kkbb[pw_id_q, pw_id_p, r, s] = np.conj(value)
            has_kkbb = True
        elif integral.type == 'kbbk':
            p, q, r, s = integral.index
            pw_id_p = _pw_id(p, integral)
            pw_id_s = _pw_id(s, integral)
            _check_ao_indices(integral, q, r)
            value = integral.value
            int2e_kbbk[pw_id_p, q, r, pw_id_s] = value
            int2e_kbbk[pw_id_s, r, q, pw_id_p] = np.conj(value)
            has_kbbk = True
        elif integral.type == 'kbkb':
            p, q, r, s = integral.index
            pw_id1 = _pw_id(p, integral)
            pw_id2 = _pw_id(r, integral)
            _check_ao_indices(integral, q, s)
            value = integral.value
            int2e_kbkb[pw_id1, q, pw_id2, s] = value
            int2e_kbkb[pw_id2, s, pw_id1, q] = value
            has_kbkb = True
        elif integral.type == 'kkkb':
            p, q, r, s = integral.index
            pw_id_p = _pw_id(p, integral)
            pw_id_q = _pw_id(q, integral)
            pw_id_r = _pw_id(r, integral)
            _check_ao_indices(integral, s)
            int2e_kkkb[pw_id_p, pw_id_q, pw_id_r, s] = integral.value
            has_kkkb = True
        elif integral.type == 'kkkk':
            p, q, r, s = integral.index
            pw_id_p = _pw_id(p, integral)
            pw_id_q = _pw_id(q, integral)
            pw_id_r = _pw_id(r, integral)
            pw_id_s = _pw_id(s, integral)
            value = integral.value
            int2e_kkkk[pw_id_p, pw_id_q, pw_id_r, pw_id_s] = value
            int2e_kkkk[pw_id_r, pw_id_s, pw_id_p, pw_id_q] = value
            int2e_kkkk[pw_id_q, pw_id_p, pw_id_s, pw_id_r] = np.conj(value)
            int2e_kkkk[pw_id_s, pw_id_r, pw_id_q, pw_id_p] = np.conj(value)
            has_kkkk = True
        else:
            raise ValueError(f"Unknown integral type: {integral.type}")

    int2e_mopw = []

    # bbbb -> bbbb transformation
    if has_bbbb:
        int2e_bbbb = np.einsum("abcd,aA,bB,cC,dD->ABCD", int2e_bbbb,
                               mo_coeff, mo_coeff, mo_coeff, mo_coeff, optimize=True)
        for p in range(mo_num):
            for q in range(p + 1):
                pq = p * (p + 1) // 2 + q
                for r in range(mo_num):
                    for s in range(r + 1):
                        rs = r * (r + 1) // 2 + s
                        if pq < rs:
                            continue
                        value = int2e_bbbb[p, q, r, s]
                        if abs(value) > cutoff:
                            int2e_mopw.append(INT2e((p, q, r, s), value, 'bbbb'))

    # kbbb -> kbbb transformation, keeping the PW leg unchanged.
    if has_kbbb:
        int2e_kbbb = np.einsum("nabc,aA,bB,cC->nABC", int2e_kbbb,
                               mo_coeff, mo_coeff, mo_coeff, optimize=True)
        for pw_id in range(pw_num):
            p = mo_num + pw_id
            for q in range(mo_num):
                for r in range(mo_num):
                    for s in range(r + 1):
                        value = int2e_kbbb[pw_id, q, r, s]
                        if abs(value) > cutoff:
                            int2e_mopw.append(INT2e((p, q, r, s), value, 'kbbb'))

    # kkbb -> kkbb transformation, keeping the PW legs unchanged.
    if has_kkbb:
        int2e_kkbb = np.einsum("pqab,aA,bB->pqAB", int2e_kkbb,
                               mo_coeff, mo_coeff, optimize=True)
        for pw_id_p in range(pw_num):
            p = mo_num + pw_id_p
            for pw_id_q in range(pw_id_p + 1):
                q = mo_num + pw_id_q
                for r in range(mo_num):
                    for s in range(r + 1):
                        value = int2e_kkbb[pw_id_p, pw_id_q, r, s]
                        if abs(value) > cutoff:
                            int2e_mopw.append(INT2e((p, q, r, s), value, 'kkbb'))

    # kbbk -> kbbk transformation, keeping the PW legs unchanged.
    if has_kbbk:
        int2e_kbbk = np.einsum("pabq,aA,bB->pABq", int2e_kbbk,
                               mo_coeff, mo_coeff, optimize=True)
        for pw_id_p in range(pw_num):
            p = mo_num + pw_id_p
            for q in range(mo_num):
                for pw_id_s in range(pw_num):
                    if pw_id_p < pw_id_s:
                        continue
                    s = mo_num + pw_id_s
                    for r in range(mo_num):
                        if pw_id_p == pw_id_s and q < r:
                            continue
                        value = int2e_kbbk[pw_id_p, q, r, pw_id_s]
                        if abs(value) > cutoff:
                            int2e_mopw.append(INT2e((p, q, r, s), value, 'kbbk'))

    # kbkb -> kbkb transformation, keeping the PW legs unchanged.
    if has_kbkb:
        int2e_kbkb = np.einsum("namb,aA,bB->nAmB", int2e_kbkb,
                                   mo_coeff, mo_coeff, optimize=True)
        for pw_id1 in range(pw_num):
            p = mo_num + pw_id1
            for q in range(mo_num):
                for pw_id2 in range(pw_num):
                    if pw_id1 < pw_id2:
                        continue
                    r = mo_num + pw_id2
                    for s in range(mo_num):
                        if pw_id1 == pw_id2 and q < s:
                            continue
                        value = int2e_kbkb[pw_id1, q, pw_id2, s]
                        if abs(value) > cutoff:
                            int2e_mopw.append(INT2e((p, q, r, s), value, 'kbkb'))

    # kkkb -> kkkb transformation, keeping the PW legs unchanged.
    if has_kkkb:
        int2e_kkkb = np.einsum("pqra,aA->pqrA", int2e_kkkb, mo_coeff, optimize=True)
        for pw_id_p in range(pw_num):
            p = mo_num + pw_id_p
            for pw_id_q in range(pw_num):
                q = mo_num + pw_id_q
                for pw_id_r in range(pw_num):
                    r = mo_num + pw_id_r
                    for s in range(mo_num):
                        value = int2e_kkkb[pw_id_p, pw_id_q, pw_id_r, s]
                        if abs(value) > cutoff:
                            int2e_mopw.append(INT2e((p, q, r, s), value, 'kkkb'))

    # kkkk -> kkkk transformation, with no AO legs to rotate.
    if has_kkkk:
        for pw_id_p in range(pw_num):
            p = mo_num + pw_id_p
            for pw_id_q in range(pw_num):
                q = mo_num + pw_id_q
                for pw_id_r in range(pw_num):
                    r = mo_num + pw_id_r
                    for pw_id_s in range(pw_num):
                        s = mo_num + pw_id_s
                        if not _is_kkkk_canonical(pw_id_p, pw_id_q, pw_id_r, pw_id_s):
                            continue
                        value = int2e_kkkk[pw_id_p, pw_id_q, pw_id_r, pw_id_s]
                        if abs(value) > cutoff:
                            int2e_mopw.append(INT2e((p, q, r, s), value, 'kkkk'))

    return int2e_mopw




def get_pqrs_id(p, q, r, s, nq, nr, ns):
    return p*nq*nr*ns + q*nr*ns + r*ns + s


def sort_int2e(int2e, norb):
    pqrs_id = []
    for i, g in enumerate(int2e):
        p, q, r, s = g.index
        nu = get_pqrs_id(p, q, r, s, norb, norb, norb)
        #^^ p, q, r, and s vary as s => r => q => p.
        kbbb_id += [nu]
    return np.argsort(pqrs_id)


def get_kbbb_id(num_ao, p, q, r, s, n_rs=None):
    '''
    Returns a composite index from a 4-number index that corresponds to a kbbb type. The
    resulting composite index starts from 1, which corresponds to p=num_ao, and q=r=s=0.

    num_ao:
      The number of AOs.
    p:
      PW index of electron 1, num_ao <= p <= num_ao+npw-1 where npw is the number of plane waves.
    q:
      AO index of electron 1, 0 <= q <= num_ao-1.
    r:
      AO index of electron 2, 0 <= r <= num_ao-1.
    s:
      AO index of electron 2, 0 <= s <= r.
    n_rs:
      The number of (r,s) pairs with r >= s. The user has the option to provide this
      argument to avoid repeatedly computing its value inside the function when it is called
      in a loop. If it is given, it must be equal to num_ao*(num_ao+1)/2.
    '''
    
    if n_rs is None:
        n_rs = round(num_ao*(num_ao+1)/2)
    pp = p - num_ao + 1
    qq = q + 1
    rr = r + 1
    ss = s + 1
    return (pp-1)*num_ao*n_rs + (qq-1)*n_rs + round((rr-1) * (1+rr-1) / 2) + ss


def sort_kbbb(int2e, num_ao):
    '''
    Sorts elements a list containing kbbb-type integrals, (P_k G_a|G_b G_c), so that
    their indices are monotonically increasing, c moving fastest and k moving slowest.
    The input list must only contain integrals where b >= c. It returns the indices that
    would sort the input list in the above manner.
    '''
    
    n_rs_pair = round(num_ao*(num_ao+1)/2)
    kbbb_id = []
    for i, g in enumerate(int2e):
        p, q, r, s = g.index    # r >= s
        nu = get_kbbb_id(num_ao, p, q, r, s, n_rs_pair)
        #^^ p, q, r, and s vary as s => r => q => p. The plane wave index, p, runs slowest.
        kbbb_id += [nu]
    return np.argsort(kbbb_id)


def int2e_ao2mo_kbbb(mo_coeff, int2e_ao, pw_num):

    '''
    Transforms kbbb 2e integrals from AO to MO representation.

    int2e_ao:
      A list of INT2e objects of the type 'kbbb', i.e (P_k G_a|G_b G_c). The elements of the
      list must be ordered so that c varies the fastest, then b, then a, then k. Only independent
      elements should be present such that k, a, and b are full range, while c<=b. Use sort_kbbb
      to ensure that int2e_ao is sorted in that manner.
    '''
    
    ao_num, mo_num = mo_coeff.shape
    dsymm = get_blas_funcs('symm', dtype='float', ilp64='preferred')
    dgemm = get_blas_funcs('gemm', dtype='float', ilp64='preferred')
    int2e_mopw = []
    mask_l = np.tril( np.ones((ao_num, ao_num), dtype=bool) )
    submat = np.zeros((ao_num, ao_num), dtype=complex)
    n_rs = round(ao_num * (ao_num+1) / 2)
    for p in range(pw_num):    # Loop over PWs
        for q in range(mo_num):      # Loop over MOs
            mat = np.zeros((mo_num, mo_num), dtype=complex)
            
            for q_ in range(ao_num):    # Loop over AOs
                i0 = p*ao_num*n_rs + q_*n_rs
                submat[mask_l] = np.array([g.value for g in int2e_ao[i0:i0+n_rs]])
                #^^ For a given k and q_, the lower triangular part of submat stores
                # (k \chi_q_ | \chi \chi) for all unique pairs of the 2nd and 3rd
                # indices.
    
                #==== Real part of submat ====#
                rtemp1 = dsymm(1.0, submat.real, mo_coeff, side=0, lower=1)
                rtemp2 = dgemm(1.0, mo_coeff, rtemp1, trans_a=1)
    
                #==== Imaginary part of submat ====#
                itemp1 = dsymm(1.0, submat.imag, mo_coeff, side=0, lower=1)
                itemp2 = dgemm(1.0, mo_coeff, itemp1, trans_a=1)
    
                mat += mo_coeff[q_,q] * (rtemp2 + 1j*itemp2)
                # (rtemp2 + 1j*itemp2) = (k \chi | \phi \phi)

            # mat = (k \phi | \phi \phi)
            for r in range(mo_num):      # Loop over MOs
                for s in range(r + 1):       # Loop over MOs
                    int2e_mopw.append(INT2e((p+mo_num, q, r, s), mat[r,s], 'kbbb'))

    return int2e_mopw
    

    
def write_FCIDUMP(filename: str, int1e: np.ndarray, int2e: list[INT2e], num_ele: int,
    core_energy: complex = 0.0 + 0.0j, cutoff: float = -1.0) -> None:
    '''
    Write the integrals to a FCIDUMP file.
    Args:
        filename: Output filename.
        int1e: One-electron integral matrix.
        int2e: Two-electron integral list.
        core_energy: Frozen-core constant energy.
        cutoff: Threshold for ignoring small integrals.
    '''
    def _canonical_int2e(integral: INT2e) -> INT2e:
        i, j, k, l = integral.index
        value = integral.value

        if integral.type == 'bbbb':
            left = (i, j) if i >= j else (j, i)
            right = (k, l) if k >= l else (l, k)
            left_pair = left[0] * (left[0] + 1) // 2 + left[1]
            right_pair = right[0] * (right[0] + 1) // 2 + right[1]
            if left_pair < right_pair:
                left, right = right, left
            return INT2e((left[0], left[1], right[0], right[1]), value, integral.type)

        if integral.type == 'kbbb':
            if k < l:
                k, l = l, k
            return INT2e((i, j, k, l), value, integral.type)

        if integral.type == 'kkbb':
            if k < l:
                k, l = l, k
            if i < j:
                i, j = j, i
                value = np.conj(value)
            return INT2e((i, j, k, l), value, integral.type)

        if integral.type == 'kbbk':
            if (i, j) < (l, k):
                return INT2e((l, k, j, i), np.conj(value), integral.type)
            return INT2e((i, j, k, l), value, integral.type)

        if integral.type == 'kbkb':
            if (i, j) < (k, l):
                return INT2e((k, l, i, j), value, integral.type)
            return INT2e((i, j, k, l), value, integral.type)

        if integral.type == 'kkkb':
            return integral

        if integral.type == 'kkkk':
            candidates = (
                ((i, j, k, l), value),
                ((k, l, i, j), value),
                ((j, i, l, k), np.conj(value)),
                ((l, k, j, i), np.conj(value)),
            )
            index, canonical_value = max(candidates, key=lambda item: item[0])
            return INT2e(index, canonical_value, integral.type)

        raise ValueError(f"Unknown integral type: {integral.type}")

    compressed_int2e = {}
    for integral in int2e:
        if integral.type not in set(ALL_INT2E_TYPES):
            raise ValueError(f"Unknown integral type: {integral.type}")
        if integral.type not in enabled_types:
            continue

        canonical = _canonical_int2e(integral)
        if not np.isfinite(canonical.value):
            raise FloatingPointError(f"Non-finite two-electron integral for {canonical.type} {canonical.index}")
        if abs(canonical.value) <= cutoff:
            continue

        key = (canonical.type, canonical.index)
        old_value = compressed_int2e.get(key)
        if old_value is not None and abs(old_value - canonical.value) > max(cutoff, 1e-10):
            raise ValueError(f"Inconsistent symmetry-equivalent integral for {canonical.type} {canonical.index}")
        compressed_int2e[key] = canonical.value

    type_rank = {integral_type: rank for rank, integral_type in enumerate(ALL_INT2E_TYPES)}
    num_basis, _ = int1e.shape
    with open(filename, "w") as f:
        f.write(f"&FCI NORB={num_basis}, NELEC={num_ele}, MS2={0}, ORBSYM={'1' * num_basis},\n")
        f.write("&END\n")

        for i in range(int1e.shape[0]):
            for j in range(i + 1):
                if abs(int1e[i, j]) > cutoff:
                    f.write(f"{int1e[i, j].real:.16E} {int1e[i, j].imag:.16E} {i + 1} {j + 1} 0 0\n")

        for (integral_type, index), value in sorted(compressed_int2e.items(),
                                                    key=lambda item: (type_rank[item[0][0]], item[0][1])):
            i, j, k, l = index
            f.write(f"{value.real:.16E} {value.imag:.16E} {i + 1} {j + 1} {k + 1} {l + 1}\n")

        if abs(core_energy) > cutoff:
            f.write(f"{core_energy.real:.16E} {core_energy.imag:.16E} 0 0 0 0\n")
            






