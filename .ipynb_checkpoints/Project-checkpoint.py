
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
    Function for constructring decaying harmonic exponential functions

    Parameters
    ----------

    IP_model: String
        Please choose "cole" or "pelton".

    kernel_index: int, float
        parameter that chances periodicity and decay rate of kernel function

    exponent: float
        number in the exponent that controls the growth (for positive values) or decay (negative values) of our kernel function

    frequency: float
        oscillation rate of our kernel functions
     """
    def __init__(self, IP_model, model_base,
        res_air, res_sea, res_seafloor, nlayers, tindex,
        resmin=1e-15, resmax=1e15, mmin = 0, mmax = 1.,
        taumin=1e-15, taumax=1e15, cmin=0, cmax=1.
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
        self.resmin = 1e-15
        self.resmax = 1e15
        self.mmin = 0
        self.mmax = 1
        self.taumin = 1e-15
        self.taumax  = 1e15
        self.cmin =0
        self.cmax =1
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

        # use initial model for reference model too
        model_old = model_init
        mref = model_init
        # get noise part
        Wd = self.get_Wd(dobs)
        # Prepare array for storing error and model
        error_prg = np.zeros(niter+1)
        model_prg = np.zeros((niter+1,model_init.shape[0]))

        r =  Wd @ (self.predicted_data(model_old) -dobs)
        phid = 0.5 * np.dot(r,r)
        phim = 0.5*  np.dot(model_old-mref,model_old-mref)
        f_old  = phid + beta*phim
        error_prg[0] = f_old
        model_prg[0,:] = model_old

        print(f'Gauss-Newton \n Initial phid = {phid:.2e} ,phim = {phim:.2e}, error= {f_old:.2e} ')
        for i in range(niter):
            # Calculate J:Jacobian and g:gradient
            J = self.Japprox(model_old)
            g = J.T @ Wd.T @ r + beta*(model_old-mref)
            H = J.T @ Wd.T @ Wd @ J + beta*np.identity(len(model_old))

            # End inversion if gradient is smaller than tolerance

            # Line search method Amijo using directional derivative
            dm = np.linalg.solve(H,g)
            g_norm = np.linalg.norm(g,ord=2)
            if  g_norm< gtol:
                print(f"Inversion complete since norm of gradient is small as :{g_norm :.3e} ")
                break
            alpha = alpha0
            model_new = self.constrain_model_vector(model_old - alpha * dm)
            r = Wd@(self.predicted_data(model_new) -dobs)
            phid = 0.5 * np.dot(r, r)
            phim = 0.5 * np.dot(model_new - mref, model_new - mref)
            f_new = phid + beta *phim
            directional_derivative = np.dot(g, -dm)
            while f_new >= f_old + alpha * mu * directional_derivative:
                alpha *= afac
                model_new = self.constrain_model_vector(model_old - alpha * dm)
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
        model_prg = model_prg[:k]
        error_prg = error_prg[:k]
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


# #
# res_air = 2e14
# res_sea = 1/3
# nlayers = 1
# #nlayers = 10
# #layer_thicknesses = 5.
# seabed_depth = 1000.5
# #seabed_depth = 1000.5
# #depth = np.hstack([np.r_[0],seabed_depth+layer_thicknesses * np.arange(nlayers)])
# depth = np.array([0,seabed_depth])
# t = np.logspace(-8,-2, 121)
# tstrt = 1e-6
# tend = 1e-2
# tindex = (t >= tstrt) & (t <= tend)
# tplot = t[tindex]
# model_base = {
#     'src':  [1.5,1.5,0,1.5,1000, 1000],
#     'rec': [0,0,1000,0,90],
#     'depth': depth,
#     'freqtime': t ,
#     'signal': 0,
#     'mrec' : True,
#     'verb': 0
# }
# #
#
#
# EMIP =  EMIP1D(model_base,res_air,res_sea,nlayers,tindex)
#
# res_0 = 1/10 * np.ones(nlayers)
# m_0 = np.log(res_0)
#
# res_8 = 1/20 * np.ones(nlayers)
# m_8 = np.log(res_8)
# #tau = [0,0, 1e-4]
# m_t = np.log(1e-4)*np.ones(nlayers)
# #c = [0,0, 0.5]
# m_c = 0.5*np.ones(nlayers)
# print(f'Chargeability for model_osb {(res_0 - res_8) / res_0}')
# model_obs = np.hstack([m_0, m_8,m_t,m_c])
# data_clean = EMIP.predicted_data(model_obs)
# relative_error=0
# np.random.seed(0)
# data_obs =  data_clean + np.random.randn(len(data_clean)) * relative_error * np.abs(data_clean)
#
# res_0 = np.ones(nlayers)
# m_0 = np.log(res_0)
#
# res_8 = np.ones(nlayers)
# m_8 = np.log(res_8)
# #tau = [0,0, 1e-4]
# m_t = np.log(1e-3)*np.ones(nlayers)
# #c = [0,0, 0.5]
# m_c = 0.6*np.ones(nlayers)
# m = (res_0 - res_8) / res_0
# print(f'chargeability for initial model{m}')
#
# model_init = np.hstack([m_0, m_8, m_t, m_c])
# data_init = EMIP.predicted_data(model_init)
# niter = 10
# beta = 1
# #model_SD, error, model_itr = EMIP.steepest_descent_linesearch(data_obs, model_init, niter)
# model_SD, error, model_itr = EMIP.steepest_descent_Reg_LS(
#     dobs=data_obs, model_init=model_init, niter=niter,beta=beta,
#     mu= 0.1
# )
#
# data_pred = (EMIP.predicted_data(model_SD))
#
# niter = 20
# beta = 1
# #model_SD, error, model_itr = EMIP.steepest_descent_linesearch(data_obs, model_init, niter)
# model_GN, error, model_itr = EMIP.GaussNewton_Reg_LS(
#     dobs=data_obs, model_init=model_SD, niter=niter,beta=beta,
#     mu= 1e-4
# )
#
