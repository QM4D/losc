import pyscf
import numpy as np
import py_losc


def _local_print(print_level, verbose_level, mol, *args):
    """Print arguments to PySCF output file if verbose_level >= print_level.

    Parameters
    ----------
    print_level : int
        The print level.
    verbose_level : int
        The verbose level.
    mol : pyscf.gto.mol.Mole
    """
    if verbose_level >= print_level:
        t = [f'{i}' for i in args]
        pyscf.lib.logger.info(mol, f' {" ".join(t)}')


def init_local_print(verbose_level, mol):
    """Create a local printer and set its verbose level

    Parameters
    ----------
    verbose_level : int
        The verbose level.
    mol: pyscf.gto.mol.Mole
    
    Returns
    -------
    local_print : function(print_level, *args)
        A printer with setting verbose level to be `verbose_level`.
    """

    def local_print(print_level, *args):
        """ Print arguments into PySCF output file when verbose level >= 
        print level.
        The verbose level is initialized at the call of 
        `pyscf_losc.utils.init_local_print()`.
        """
        _local_print(print_level, verbose_level, mol, *args)

    return local_print


def print_total_energies(verbose_level, mol, losc_data,  
                         print_level=1):
    """Print total energies. Default print_level=1.
    """
    local_print = init_local_print(verbose_level, mol)
    l = print_level
    local_print(l, f' ==> {losc_data["losc_type"]} Energetics <==')
    local_print(l, 'DFA energy: {:.16f}'.format(losc_data['dfa_energy']))
    local_print(l, 'LOSC correction energy: {:.16f}'.format(
        losc_data['losc_energy']
    ))
    local_print(
        l, 'LOSC-DFA total energy: {:.16f}'.format(
            losc_data['losc_dfa_energy']
        )
    )


def print_orbital_energies(verbose_level, mf, losc_data, print_level=1,
                           window=None):
    """Print orbital energies.
    """
    local_print = init_local_print(verbose_level, mf.mol)
    nspin = losc_data['nspin']
    orbital_energy_unit = losc_data['orbital_energy_unit']
    dfa_eigs = losc_data['dfa_orbital_energy']
    losc_eigs = losc_data['losc_dfa_orbital_energy']

    nao = mf.mol.nao_nr()
    is_rks = True if nspin == 1 else False
    _, occ_idx, occ_val = form_occ(mf, losc_data['occ'])
    occ_idx_val = [dict(zip(occ_idx[s], occ_val[s])) for s in range(nspin)]
    l = print_level
    local_print(
        l, f' ==> {losc_data["losc_type"]} Orbital Energies <=='
    )
    for s in range(nspin):
        if not is_rks:
            local_print(
                l, f"{'Alpha' if s == 0 else 'Beta'} orbital energy:"
            )
        local_print(
            l, "{:<5s}  {:<8s}  {:>14s} {:>14s}"
            .format("Index", "Occ", f"DFA ({orbital_energy_unit})",
                    f"LOSC ({orbital_energy_unit})")
        )
        if nao != (dfa_eigs[s].shape)[0]:
            local_print(
                l, f"Warning: spin={s} number of basis NOT equal to number of AOs."
            )
        for i, orbE in enumerate(dfa_eigs[s]):
            #if orbital_energy_unit != 'eV':
            #    orb *= constants.hartree2ev
            if not window or (window[0] <= orbE <= window[1]):
                local_print(
                    l, "{:<5d}  {:<8.5f} {:>14.6f}  {:>14.10f}"
                    .format(
                        i, occ_idx_val[s].get(i, 0),
                        dfa_eigs[s][i], losc_eigs[s][i]
                    )
                )


def print_full_matrix(mat, mol, line_limit=5):
    """Print matrix into PySCF output.
    """
    pyscf.lib.logger.info(mol, f'Dimension: {mat.shape}')
    row, col = mat.shape
    for i in range(row):
        pyscf.lib.logger.info(mol, f'    {i}:')
        pyscf.lib.logger.info(mol, f' {mat[i]}')


def print_sym_matrix(mat, mol, line_limit=5):
    """Print matrix into PySCF output
    """
    pyscf.lib.logger.info(mol, f'Dimension: {mat.shape}')
    row, col = mat.shape
    for i in range(row):
        pyscf.lib.logger.info(mol, f'    {i}:')
        pyscf.lib.logger.info(mol, f' {mat[i][:i+1]}')


def calc_orbitalet_e(mo_energy, U):
    sqrd_U = U **2
    orbitalet_e = np.dot(sqrd_U, mo_energy)
    return orbitalet_e


def sort_orbitalets(orbitalet_e, lo_coeff):
    idx = orbitalet_e.argsort()
    orbitalet_e = orbitalet_e[idx]
    lo_coeff = lo_coeff[:,idx]
    return orbitalet_e, lo_coeff


def form_occ(mf, occ={}):
    """Form the occupation number of a given pyscf.dft.rks.RKS or 
    pyscf.dft.uks.UKS object by modifying mf.mo_occ from the parent DFA 
    calculation. 

    Parameters
    ----------
    mf : pyscf.dft.rks.RKS or pyscf.dft.uks.UKS
        Restricted or Unrestricted KS SCF object
    occ : dict, default to an empty dict.
        A dictionary that specifies the customized occupancy.

        The structure of the dict:
        {
            "alpha": # for alpha spin
            {
                int or str: float,
            }
            "beta": # for beta spin
            {
                int or str: float,
            }
        }
        - int: float    int refers to orbital index (0-based);
                        float refers to the customized occupation number (in
                        the range of [0, 1]).
        - str: float    str can be either 'homo' or 'lumo';
                        float refers to the customized occupation number (in
                        the range of [0, 1])

    Returns
    -------
    nocc : [int, int]
        Number of non-zero occupied orbitals.
    occ_idx : [[int, ...], [int, ...]]
        The index (0-based) of non-zero occupied orbitals.
    occ_val : [[float, ...], [float, ...]]
        The occupation number of non-zero occupied orbitals.
    """
    
    # step 1: build aufbau occupation based on mf.
    # If mf is a pyscf.dft.rks.RKS object, mf.mo_occ is a one-dimensional 
    # numpy.array object. [float ...]
    # If mf is a pyscf.dft.uks.UKS object, mf.mo_occ is a two-dimensional
    # numpy.array object. [[float ...]
    #                      [float ...]]

    nelec = mf.mol.nelec
    rst_occ = [{i: 1 for i in range(n)} for n in nelec]

    # step 2: update occupation number
    spin_chanel = ['alpha', 'beta']
    for spin_id, occ_info in occ.items():
        spin_id = spin_id.lower()
        if spin_id not in spin_chanel:
            raise Exception(
                f"Invalid customized occupation spin chanel: {spin_id}."
            )
        s = spin_chanel.index(spin_id)
        homo_status = []
        lumo_status = []
        for orb_i, occ_i in occ_info.items():
            # check the type of orbital index
            if isinstance(orb_i, str):
                if orb_i not in ['homo', 'lumo']:
                    raise Exception(
                        f"Invalid customized occupation index: {orb_i}."
                    )
                if orb_i == 'homo':
                    orb_i = nelec[s] - 1
                else:
                    orb_i = nelec[s]
            elif not isinstance(orb_i, int):
                raise Exception(
                    "Orbital index should be int, 'homo', or 'lumo'."
                )
            if not 0 <= orb_i < mf.mo_coeff.shape[1]:
                raise Exception(
                    "Customized occupation index is out-of-range."
                )
            # check if homo and homo idx appear at the same time.
            if orb_i == nelec[s] - 1:
                homo_status.append(orb_i)
            if len(homo_status) == 2:
                raise Exception(
                    "HOMO and HOMO index appear at the same time."
                )
            # check if lumo and lumo idx appear at the same time.
            if orb_i == nelec[s]:
                lumo_status.append(orb_i)
            if len(lumo_status) == 2:
                raise Exception(
                    "LUMO and LUMO index appear at the same time."
                )
            # update the occ number
            rst_occ[s][orb_i] = occ_i
    
    occ_idx = []
    occ_val = []
    for d in rst_occ:
        if d:
            idx_occ = list(d.items())
            # remove 0 occupied orbitals
            idx_occ = list(filter(lambda x: abs(x[1]) > 0, idx_occ))
            idx_occ.sort()
            if idx_occ:
                idx, occ = zip(*idx_occ)
            else:
                idx, occ = tuple(), tuple()
            occ_idx.append(idx)
            occ_val.append(occ)
        else:
            occ_idx.append(tuple())
            occ_val.append(tuple())

    nocc = tuple(len(x) for x in occ_idx)

    return nocc, occ_idx, occ_val


def form_df_matrix(mf, C_lo, df_basis=None):
    """Build density fitting related matrices, df_pii, df_Vpq_inv

    Parameters
    ----------
    mf : pyscf.dft.rks.RKS or pyscf.dft.uks.UKS
        Wavefunction object
    C_lo : [numpy.array, ...]
        LOSC LO coefficient matrix.

    Returns
    -------
    df_pii : [numpy.array, ...]
        The three center <fitbasis|1/r|lo, lo>
    df_Vpq_inv :
        The inverse of <fitbasis|1/r|fitbasis> matrix.
    """
    nspin = len(C_lo)
    # step 1: build an auxiliary pyscf.gto.Mole object for DF.
    auxmol = pyscf.df.addons.make_auxmol(mf.mol, df_basis)
    # step 2: compute df_pii
    # step 2.1: compute 3-center-2-electron ao integrals <fitbasis|lo, lo>.
    df_pmn = pyscf.df.incore.aux_e1(mf.mol, auxmol, intor='int3c2e', 
                                    aosym='s1', comp=None, out=None)
    #########################################################################
    # In PySCF, this matrix is stored in an anti-intuitive way. The first   #
    # index is AO, the second index is fitbasis, and the third index is AO. #
    # No idea why the matrix is strored like this.                          #
    #########################################################################
    # step 2.2: transform from AO to LO. 
    df_pii = [
        np.zeros((auxmol.nao_nr(), C_lo[s].shape[1])) for s in range(nspin)
    ]
    for s in range(nspin):
        df_pii[s] = np.einsum(
            'mi,mpn,ni->pi', C_lo[s], df_pmn, C_lo[s], optimize=True
        )

    # step 3: compute df_Vpq_inv.
    df_Vpq = pyscf.df.incore.fill_2c2e(mf.mol, auxmol, intor='int2c2e', 
                                       comp=None, hermi=1, out=None)
    df_Vpq_inv = np.linalg.inv(df_Vpq)

    return df_pii, df_Vpq_inv


def generate_loscmf(mf, losc_data=None):
    """This function is used to generate an instance of SCF class in 
    PySCF for SCF-LOSC calculation.

    Parameters
    ----------
    mf : pyscf.dft.rks.RKS or pyscf.dft.uks.UKS
        mf object holds all parameters to control the SCF calculation
        in PySCF. The input mf object should control the parent DFA
        calculation in PySCF.
        The member functions that will be modified for LOSC calculation
        are
        - mf.get_fock
        - mf.energy_tot
    losc_data : dict
        A dictionary that holds useful matrices and information from 
        post-SCF-LOSC calculation.

    Returns
    -------
    loscmf : pyscf.dft.rks.RKS or pyscf.dft.uks.UKS
        A new object with the member functions .get_fock and .energy_tot
        modified for SCF-LOSC calculation.
    """
    # make a copy of the input mf object
    loscmf = mf
    self = mf
    original_get_fock = mf.get_fock
    original_energy_tot = mf.energy_tot
    # determine nspin
    if isinstance(mf, pyscf.dft.rks.RKS):
        nspin = 1
    else:
        nspin = 2
    # determine if overwriting the member functions is necessary
    if losc_data == None:
        return loscmf
    else:
        curvature = losc_data.get('curvature', [])
        C_lo = losc_data.get('C_lo', [])
        E_losc = [0]

        def get_fock(h1e=None, s1e=None, vhf=None, dm=None, cycle=-1,
                     diis=None, diis_start_cycle=None, 
                     level_shift_factor=None, damp_factor=None):
            ''' overwrite Fock matrix
            '''
            # ovlp matrix
            S = np.asarray(loscmf.mol.intor('int1e_ovlp'))
            # density matrix
            if nspin == 1:
                total_D = mf.make_rdm1(loscmf.mo_coeff, loscmf.mo_occ)
                alpha_D = [0.5 * val for val in total_D]
                D = [np.asarray(alpha_D), np.asarray(alpha_D)] 
            else:
                D = [
                    np.asarray(loscmf.make_rdm1(loscmf.mo_coeff, loscmf.mo_occ)[0]),
                    np.asarray(loscmf.make_rdm1(loscmf.mo_coeff, loscmf.mo_occ)[1])
                ]
            # Fock matrix
            if nspin == 1:
                F = [
                    np.asarray(original_get_fock()), 
                    np.asarray(original_get_fock())]
            else:
                F = [
                    np.asarray(original_get_fock()[0]),
                    np.asarray(original_get_fock()[1])
                ]
            E_losc[0] = 0
            for s in range(nspin):
                # local occupation matrix
                local_occ = py_losc.local_occupation(C_lo[s], S, D[s])
                # LOSC effective Fock matrix
                H_losc = py_losc.ao_hamiltonian_correction(
                    S, C_lo[s], curvature[s], local_occ
                )
                F[s][:] += H_losc
                # LOSC energy correction
                E_losc[0] += py_losc.energy_correction(
                    curvature[s], local_occ
                )
            if nspin == 1:
                E_losc[0] *= 2
            return F

        def energy_tot(dm=None, h1e=None, vhf=None):
            E_dfa = original_energy_tot()
            E_tot = E_dfa + E_losc[0]
            return E_tot

        loscmf.get_fock = get_fock
        loscmf.energy_tot = energy_tot

    return loscmf


def form_grid_lo(mf, C_lo):
    """Form the matrix that stores the values of LOs on grid points.

    Parameters
    ----------
    mf : pyscf.dft.rks.RKS or pyscf.dft.uks.UKS
        A PySCF object associated with DFT calculations.
    C_lo : np.array [nbasis, nlo]
        LO coefficient matrix on AOs.

    Returns
    -------
    grid_lo : np.array [npts, nlo]
        A matrix for the values of LOs on grid points.
        npt: the number of grid points.
        nlo: the number of LOs.

    """
    # step 1: sanity-check
    if not isinstance(mf, pyscf.dft.rks.RKS):
        if not isinstance(mf, pyscf.dft.uks.UKS):
            raise Exception(
                f"Unknown type of arguement {mf}."
            )

    # step 2: read grid coordinates mf.grids.coords
    grid_coords = mf.grids.coords
    npts = grid_coords.shape[0]

    # step 3: use pyscf.dft.numint.eval_ao(mf.mol, coords, ...) to evaluate 
    #         AO function values on the given grids.
    grid_ao = pyscf.dft.numint.eval_ao(mf.mol, grid_coords)
    # This will return a 2D array of shape (npts, nao), where npts is the 
    # number of grids and nao is the number of AOs.

    # step 4: use C_lo to convert AO values in step 3 to LO values.
    nlo = C_lo.shape[1]
    grid_lo = np.zeros((npts, nlo))
    grid_lo = grid_ao.dot(C_lo)
    
    return grid_lo
