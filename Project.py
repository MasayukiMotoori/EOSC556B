
import pandas as pd
from scipy import optimize
from scipy.constants import mu_0, epsilon_0

import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.linalg import lu_factor, lu_solve
import empymod


class EMIP1D:
    """
    Class for one dimensional inversion for about Induced Polarization from
    Time-Domain Electromagnetic about Sea Massive Sulfide Exploration.
    Forward modelling: empymod
    https://empymod.emsig.xyz/en/stable/index.html

    Forward modelling: SimPEG

    Inversion
        Jacobian is approximated by finite difference
        Objective function: f = 0.5*( phid + beta* phim)
        Data part: phid = (Wd(F(m)-dobs))**2
        Model part: phim = (Ws(mref-minit))**2

    Parameters

    IP_model: String
        "cole":
            res0: resistivity in low frequency
            res8: resistivity in hig frequency
            tau : time constant
            c   : relaxation parameter

        "pelton":
            res : resistivity
            m   : chargeability
            tau : time constant
            c   : relaxation parameter


    res_air: float
        resistivity of air

    res_sea: float
        resistivity of sea

    res_seafloor: float
        resistivity of seafloor, background

    nlayers: integer
        number of layers

    tindex: boolen
        time index to use for forward modelling and inversion

    model_base: please refer empymod tutorial
    https://empymod.emsig.xyz/en/stable/api/empymod.model.dipole.html
        'src':  transmitter configuration
        'rec': receiver configuration
        'depth': depth,
        'freqtime': t as time domain
        'signal': ssingal wave
        'mrec' : True
        'verb': 0
     """
    def __init__(self, IP_model, model_base,
        res_air, res_sea, res_seafloor, nlayers, tindex,
        resmin=1e-10, resmax=1e10, mmin = 0, mmax = 1.,
        taumin=1e-10, taumax=1e10, cmin=0, cmax=1.
        ):
        self.IP_model = IP_model
        self.model_base = model_base
        self.res_air = res_air
        self.res_sea = res_sea
        self.res_seafloor = res_seafloor
        self.nlayers = nlayers
        self.tindex  = tindex
        self.resmin = resmin
        self.resmax = resmax
        self.mmin = mmin
        self.mmax = mmax
        self.taumin = taumin
        self.taumax  = taumax
        self.cmin =cmin
        self.cmax =cmax

    def cole_cole(self,inp, p_dict):
        """Cole and Cole (1941)."""

        # Compute complex conductivity from Cole-Cole
        iotc = np.outer(2j*np.pi*p_dict['freq'], inp['tau'])**inp['c']
        condH = inp['cond_8'] + (inp['cond_0']-inp['cond_8'])/(1+iotc)
        condV = condH/p_dict['aniso']**2

        # Add electric permittivity contribution
        etaH = condH + 1j*p_dict['etaH'].imag
        etaV = condV + 1j*p_dict['etaV'].imag
        PA_plot = 0
        if PA_plot == 1:
            freq = p_dict['freq']
            amplitude = np.abs(1/condH)[:,1]
            phase = np.angle(1/condH)[:,1]
        #    m_colecole = (inp['cond_8'][2]-inp['cond_0'][2])/inp['cond_8'][2]

            csv_colecole = np.array([freq,amplitude,phase]).T
            with open('EOSC555_Rep_PA_ColeCole.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['freq', 'amplitude', 'phase'])  # Write header row
                writer.writerows(csv_colecole)

        return etaH, etaV

    def pelton_et_al(self,inp, p_dict):
        """ Pelton et al. (1978)."""

        # Compute complex resistivity from Pelton et al.
        iotc = np.outer(2j * np.pi * p_dict['freq'], inp['tau']) ** inp['c']
        rhoH = inp['rho_0'] * (1 - inp['m'] * (1 - 1 / (1 + iotc)))
        rhoV = rhoH * p_dict['aniso'] ** 2

        # Add electric permittivity contribution
        etaH = 1 / rhoH + 1j * p_dict['etaH'].imag
        etaV = 1 / rhoV + 1j * p_dict['etaV'].imag

        PA_plot = 0
        if PA_plot == 1:
            freq = p_dict['freq']
            amplitude = np.abs(rhoH)[:, 2]
            phase = np.angle(rhoH)[:, 2]

            csv_colecole = np.array([freq, amplitude, phase]).T
            with open('EOSC555_Rep_PA_Pelton.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['freq', 'amplitude', 'phase'])  # Write header row
                writer.writerows(csv_colecole)

        return etaH, etaV

    def ip_model(self,model_vector):
        if self.IP_model == "cole":
            res_0 = np.hstack([[self.res_air, self.res_sea],
                    np.exp(model_vector[:self.nlayers]), [self.res_seafloor]])
            res_8 = np.hstack([[self.res_air, self.res_sea],
                    np.exp(model_vector[self.nlayers:2 * self.nlayers]), [self.res_seafloor]])
            tau = np.hstack([[1e-3, 1e-3],
                  np.exp(model_vector[2 * self.nlayers:3 * self.nlayers]), 1e-3])
            c = np.hstack([[0., 0.], model_vector[3 * self.nlayers:4 * self.nlayers],[0.]])
            cole_model = {'res': res_0, 'cond_0': 1 / res_0, 'cond_8': 1 / res_8,
                          'tau': tau, 'c': c, 'func_eta': self.cole_cole}
            return cole_model
        if self.IP_model == "pelton":
            res_0 = np.hstack([[self.res_air, self.res_sea],
                    np.exp(model_vector[:self.nlayers]), [self.res_seafloor]])
            m = np.hstack([[0., 0.], model_vector[self.nlayers:2 * self.nlayers],[0.]])
            tau = np.hstack([[1e-3, 1e-3],
                  np.exp(model_vector[2 * self.nlayers:3 * self.nlayers]), 1e-3])
            c = np.hstack([[0., 0.], model_vector[3 * self.nlayers:4 * self.nlayers],[0.]])
            pelton_model = {'res': res_0, 'rho_0': res_0, 'm': m,
                            'tau': tau, 'c': c, 'func_eta': self.pelton_et_al}

            return pelton_model

    def plot_model(self, model, ax, name, color):
        depth = self.model_base["depth"]
        depth_plot = np.vstack([depth, depth]).flatten(order="F")[1:]
        depth_plot = np.hstack([depth_plot, depth_plot[-1] * 1.5])
        model_plot = np.vstack([model, model]).flatten(order="F")[2:]
        return ax.plot(model_plot, depth_plot, color, label=name)

    def predicted_data(self, model_vector):
        if self.IP_model == "cole":
            res_0 = np.hstack([[self.res_air, self.res_sea],
                    np.exp(model_vector[:self.nlayers]), [self.res_seafloor]])
            res_8 = np.hstack([[self.res_air, self.res_sea],
                    np.exp(model_vector[self.nlayers:2 * self.nlayers]), [self.res_seafloor]])
            tau = np.hstack([[1e-3, 1e-3],
                  np.exp(model_vector[2 * self.nlayers:3 * self.nlayers]), 1e-3])
            c = np.hstack([[0., 0.], model_vector[3 * self.nlayers:4 * self.nlayers],[0.]])
            cole_model = {'res': res_0, 'cond_0': 1 / res_0, 'cond_8': 1 / res_8,
                          'tau': tau, 'c': c, 'func_eta': self.cole_cole}
            data = empymod.bipole(res=cole_model, **self.model_base)
        if self.IP_model == "pelton":
            res_0 = np.hstack([[self.res_air, self.res_sea],
                    np.exp(model_vector[:self.nlayers]), [self.res_seafloor]])
            m = np.hstack([[0., 0.], model_vector[self.nlayers:2 * self.nlayers],[0.]])
            tau = np.hstack([[1e-3, 1e-3],
                  np.exp(model_vector[2 * self.nlayers:3 * self.nlayers]), 1e-3])
            c = np.hstack([[0., 0.], model_vector[3 * self.nlayers:4 * self.nlayers],[0.]])
            pelton_model = {'res': res_0, 'rho_0': res_0, 'm': m,
                            'tau': tau, 'c': c, 'func_eta': self.pelton_et_al}

            data = empymod.bipole(res=pelton_model, **self.model_base)
        return np.array(data)[self.tindex]


    def constrain_model_vector(self,model_vector):
        "Parametert Projection based on provided bound information"

        model_vector[:self.nlayers] = np.clip(
            model_vector[:self.nlayers], np.log(self.resmin), np.log(self.resmax))
        if self.IP_model == "cole":
            for i in range(self.nlayers):
                resmaxi = model_vector[i]
                model_vector[self.nlayers + i] = np.clip(model_vector[
                        self.nlayers + i],np.log(self.resmin),self.resmax)
        else:
            model_vector[self.nlayers:2*self.nlayers] = np.clip(
            model_vector[self.nlayers:2*self.nlayers], self.mmin, self.mmax)
        model_vector[2 * self.nlayers:3 * self.nlayers] = np.clip(
            model_vector[2 * self.nlayers:3 * self.nlayers], np.log(self.taumin), np.log(self.taumax))
        model_vector[3 * self.nlayers:4 * self.nlayers] = np.clip(
            model_vector[3 * self.nlayers:4 * self.nlayers], self.cmin, self.cmax)
        return model_vector

    def Japprox(self,model_vector, perturbation=0.1, min_perturbation=1e-3):
        """"
        Jacobian Approximation using finite difference
        parameter
        model_vector: model parameter to approximate Jacobian
        perturbation: delta m

        Output
        Jacobian:
        """
        delta_m = min_perturbation  # np.max([perturbation*m.mean(), min_perturbation])
        J = []

        for i, entry in enumerate(model_vector):
            mpos = model_vector.copy()
            mpos[i] = entry + delta_m

            mneg = model_vector.copy()
            mneg[i] = entry - delta_m

            pos = self.predicted_data(self.constrain_model_vector(mpos))
            neg = self.predicted_data(self.constrain_model_vector(mneg))
            J.append((pos - neg) / (2. * delta_m))

        return np.vstack(J).T

    def get_Wd(self, dobs, ratio=0.01, plateau=1e-4):
        return np.diag(1 / (np.abs(ratio*dobs) + plateau) )

    def get_Ws(self):
        return

    def steepest_descent(self,dobs, model_init, niter):
        model_vector = model_init
        r =  dobs -self.predicted_data(model_vector)
        f = 0.5*np.dot(r,r)

        error = np.zeros(niter+1)
        error[0] = f
        model_itr = np.zeros((niter+1,model_vector.shape[0]))
        model_itr[0,:] = model_vector

        print(f'Steepest Descent \n initial phid= {f:.3e} ')
        for i in range(niter):
            J = self.Japprox(model_vector)
            r =  dobs - self.predicted_data(model_vector)
            dm =   J.T @ r
            g = np.dot(J.T, r)
            Ag = J @ g
            alpha =  np.mean(Ag * r) / np.mean(Ag * Ag)
            model_vector =  self.constrain_model_vector(model_vector + alpha * dm)
            r = self.predicted_data(model_vector) - dobs
            f = 0.5 * np.dot(r, r)
            if np.linalg.norm(dm) < 1e-12:
                break
            error[i+1] = f
            model_itr[i+1, :] = model_vector
            print(f' i= {i:3d}, phid= {f:.3e} ')
        return model_vector, error, model_itr

    def steepest_descent_Reg_LS(self,dobs, model_init, niter, beta,
        alpha0=1, afac=0.5, atol=1e-6, gtol=1e-3, mu=1e-4):
        """""
        Steepest descent
        Line search method Amijo using directional derivative

        parameter
            dobs: data
            model_init: initial model
            mref : applly initial model as reference model
            niter: max iteration number
            beta: beta for model part
            alpha0: initial alpha for line search
            afac: backtracking factor
            atol: min value for alpha
            gtol: minimum value for dradient, stopping criteria for inversion
            mu: parameter for directional derivative
        """

        model_old = model_init
        mref = model_init
        Wd = self.get_Wd(dobs)
        error_prg = np.zeros(niter+1)
        model_prg = np.zeros((niter+1,model_init.shape[0]))
        r =  Wd @ (self.predicted_data(model_old) -dobs)
        phid = 0.5 * np.dot(r,r)
        phim = 0.5*  np.dot(model_old-mref,model_old-mref)
        f_old  = phid + beta*phim
        error_prg[0] = f_old
        model_prg[0,:] = model_old
        print(f'Steepest Descent \n Initial phid = {phid:.2e} ,phim = {phim:.2e}, error= {f_old:.2e} ')
        for i in range(niter):
            # Calculate J:Jacobian and g:gradient
            J = self.Japprox(model_old)
            g =  J.T @ Wd.T @ r+ beta*(model_old-mref)
            # End inversion if gradient is smaller than tolerance
            g_norm = np.linalg.norm(g, ord=2)
            if  g_norm< gtol:
                print(f"Inversion complete since norm of gradient is small as :{g_norm :.3e} ")
                break

            # Line search method Amijo using directional derivative
            alpha = alpha0
            directional_derivative = np.dot(g, -g)

            model_new = self.constrain_model_vector(model_old - alpha * g)
            r = Wd@(self.predicted_data(model_new) -dobs)
            phid = 0.5 * np.dot(r, r)
            phim = 0.5 * np.dot(model_new - mref, model_new - mref)
            f_new = phid + beta *phim
            while f_new >= f_old + alpha * mu * directional_derivative:
                alpha *= afac
                model_new = self.constrain_model_vector(model_old - alpha * g)
                r = Wd@(self.predicted_data(model_new) - dobs)
                phid = 0.5 * np.dot(r, r)
                phim = 0.5 * np.dot(model_new - mref, model_new - mref)
                f_new = phid +  beta * phim
                if np.linalg.norm(alpha) < atol:
                    break
            model_old = model_new
            model_prg[i+1, :] = model_new
            f_old = f_new
            error_prg[i+1] = f_new
            k = i+1
            print(f'{k:3}, alpha:{alpha:.2e}, gradient:{g_norm:.2e}, phid:{phid:.2e}, phim:{phim:.2e}, f:{f_new:.2e} ')
        # filter model prog data
        model_prg = model_prg[:k]
        error_prg = error_prg[:k]
        return model_new,error_prg,model_prg

    def GaussNewton_Reg_LS(self,dobs, model_init, niter, beta,
        alpha0=1, afac=0.5, atol=1e-6, gtol=1e-3, mu=1e-4):
        """""
        Gauss-Newton method
        Line search method Amijo using directional derivative

        parameter
            dobs: data
            model_init: initial model
            mref : applly initial model as reference model
            niter: max iteration number
            beta: beta for model part
            alpha0: initial alpha for line search
            afac: backtracking factor
            atol: min value for alpha
            gtol: minimum value for dradient, stopping criteria for inversion
            mu: parameter for directional derivative
        """

        model_old = model_init
        # applay initial model for reference mode
        mref = model_init
        # get noise part
        Wd = self.get_Wd(dobs)
        # Initialize object function
        r =  Wd @ (self.predicted_data(model_old) -dobs)
        phid = 0.5 * np.dot(r,r)
        phim = 0.5*  np.dot(model_old-mref,model_old-mref)
        f_old  = phid + beta*phim
        # Prepare array for storing error and model in progress
        error_prg = np.zeros(niter+1)
        model_prg = np.zeros((niter+1,model_init.shape[0]))
        error_prg[0] = f_old
        model_prg[0,:] = model_old

        print(f'Gauss-Newton \n Initial phid = {phid:.2e} ,phim = {phim:.2e}, error= {f_old:.2e} ')
        for i in range(niter):

            # Jacobian
            J = self.Japprox(model_old)

            # gradient
            g = J.T @ Wd.T @ r + beta*(model_old-mref)

            # Hessian approximation
            H = J.T @ Wd.T @ Wd @ J + beta*np.identity(len(model_old))

            # model step
            dm = np.linalg.solve(H,g)

            # End inversion if gradient is smaller than tolerance
            g_norm = np.linalg.norm(g,ord=2)
            if  g_norm< gtol:
                print(f"Inversion complete since norm of gradient is small as :{g_norm :.3e} ")
                break

            # update object function
            alpha = alpha0
            model_new = self.constrain_model_vector(model_old - alpha * dm)
            r = Wd@(self.predicted_data(model_new) -dobs)
            phid = 0.5 * np.dot(r, r)
            phim = 0.5 * np.dot(model_new - mref, model_new - mref)
            f_new = phid + beta *phim

            # Backtracking method using directional derivative Amijo
            directional_derivative = np.dot(g, -dm)
            while f_new >= f_old + alpha * mu * directional_derivative:
                # backtracking
                alpha *= afac
                # update object function
                model_new = self.constrain_model_vector(model_old - alpha * dm)
                r = Wd@(self.predicted_data(model_new) - dobs)
                phid = 0.5 * np.dot(r, r)
                phim = 0.5 * np.dot(model_new - mref, model_new - mref)
                f_new = phid +  beta * phim
                # Stopping criteria for backtrackinng
                if alpha < atol:
                    break

            # Update model
            model_old = model_new
            model_prg[i+1, :] = model_new
            f_old = f_new
            error_prg[i+1] = f_new
            k = i+1
            print(f'{k:3}, alpha:{alpha:.2e}, gradient:{g_norm:.2e}, phid:{phid:.2e}, phim:{phim:.2e}, f:{f_new:.2e} ')
        # clip progress of model and error in inversion
        error_prg = error_prg[:k]
        model_prg = model_prg[:k]
        return model_new,error_prg,model_prg


    def get_r08_grid (self, dobs, m_0 ,m_8, m_t,m_c,beta, mref, ngrid=20, mirgin = 0.1, null_value=-1):
        # get grid of object function for the range of given model parameter
        m0_grid = np.linspace(np.min(m_0) - mirgin, np.max(m_0) + mirgin, ngrid)
        m8_grid = np.linspace(np.min(m_8) - mirgin, np.max(m_8) + mirgin, ngrid)
        cc_grid = np.zeros((ngrid, ngrid))
        Wd = self.get_Wd(dobs)

        for j, m0_tmp in enumerate(m0_grid):
            for i, m8_tmp in enumerate(m8_grid):
                if m8_tmp <= m0_tmp:
                    model_vector = np.hstack([m0_tmp, m8_tmp, m_t, m_c])
                    r = Wd @ (self.predicted_data(model_vector) - dobs)
                    phid = 0.5 * np.dot(r, r)
                    phim = 0.5 * np.dot(model_vector - mref, model_vector - mref)
                    cc_grid[i, j] = phid + beta*phim
                else:
                    cc_grid[i,j] = null_value

        cc_grid = np.ma.masked_values(cc_grid, null_value)
        return m0_grid, m8_grid, cc_grid

    def get_rm_grid (self, dobs, m_r ,m_m, m_t,m_c,beta, mref, ngrid=20, mirgin = 0.1):
        # get grid of object function for the range of given model parameter
        mr_grid = np.linspace(np.min(m_r) - mirgin, np.max(m_r) + mirgin, ngrid)
        mm_grid = np.linspace(np.min(m_m) - mirgin, np.max(m_m) + mirgin, ngrid)
        cc_grid = np.zeros((ngrid, ngrid))
        Wd = self.get_Wd(dobs)

        for j, mr_tmp in enumerate(mr_grid):
            for i, mm_tmp in enumerate(mm_grid):
                model_vector = np.hstack([mr_tmp, mm_tmp, m_t, m_c])
                r = Wd @ (self.predicted_data(model_vector) - dobs)
                phid = 0.5 * np.dot(r, r)
                phim = 0.5 * np.dot(model_vector - mref, model_vector - mref)
                cc_grid[i, j] = phid + beta*phim
        return mr_grid, mm_grid, cc_grid

#
# res_air = 2e14
# res_sea = 1/3
# res_seafloor = 1
# nlayers = 5
#
# layer_thicknesses = 5.
# seabed_depth = 1000.1
# depth = np.hstack([np.r_[0],seabed_depth+layer_thicknesses * np.arange(nlayers+1)])
# t = np.logspace(-4,-2, 21)
# tstrt = 1e-4
# tend = 1e-2
# tindex = (t >= tstrt) & (t <= tend)
# tplot = t[tindex]
#
#
# model_base = {
#     'src':  [1.75,1.75,-1.75,1.75,1000, 1000],
#     'rec': [0,0,1000,0,90],
#     'depth': depth,
#     'freqtime': t ,
#     'signal': 0,
#     'mrec' : True,
#     'verb': 0
# }
#
# EMIP =  EMIP1D("pelton",model_base,
#     res_air,res_sea,res_seafloor,nlayers,tindex)
#
# res = 1/10 * np.ones(nlayers)
# m_r = np.log(res)
#
# m = 0.6 * np.ones(nlayers)
# m_m = m
#
# m_t = np.log(1e-4)*np.ones(nlayers)
#
# m_c = 0.5*np.ones(nlayers)
#
# model_obs = np.hstack([m_r, m_m,m_t,m_c])
# data_clean = EMIP.predicted_data(model_obs)
#
# relative_error=0.05
# np.random.seed(0)
# data_obs =  data_clean + np.random.randn(len(data_clean)) * relative_error * np.abs(data_clean)
#
# model_ip =  EMIP.ip_model(model_obs)
#
# fig, ax = plt.subplots(1, 2)
#
# # plot_model_m(model_base["depth"], model_ip["res"], ax[0], "resistivity","k")
# EMIP.plot_model(model_ip["res"], ax[0], "resistivity", "k")
# ax[0].set_ylim([1100, 900])
# ax[1].loglog(t, data_obs, "-", color="C0", label="data")
# ax[1].loglog(t, -data_obs, "--", color="C0")
#
# for a in ax:
#     a.legend()
#     a.grid()
#
# plt.tight_layout()
