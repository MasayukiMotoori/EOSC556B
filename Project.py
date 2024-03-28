
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

    def __init__(self, model_base, res_air, res_sea, nlayers,tindex):
        self.model_base = model_base
        self.res_air = res_air
        self.res_sea = res_sea
        self.nlayers = nlayers
        self.tindex  = tindex

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

    def get_cole_model(self,model_vector):
        # start by assuming our model is log(res_0), log(res_8)
        res_0 = np.hstack([[self.res_air, self.res_sea], np.exp(model_vector[:self.nlayers])])
        res_8 = np.hstack([[self.res_air, self.res_sea], np.exp(model_vector[self.nlayers:2 * self.nlayers])])
        tau = np.hstack([[0, 0], np.exp(model_vector[2*self.nlayers:3*self.nlayers])])
        c = np.hstack([[0, 0], model_vector[3*self.nlayers:4*self.nlayers]])
        m = (res_0 - res_8) / res_0
        cole_model = {'res': res_0, 'cond_0': 1 / res_0, 'cond_8': 1 / res_8,
                      'tau': tau, 'c': c, 'func_eta': self.cole_cole}
        return cole_model

    def predicted_data(self, model_vector):
        cole_model = self.get_cole_model(model_vector)
        data_colecole = empymod.bipole(res=cole_model, **self.model_base)
     #   data_pelton = empymod.bipole(res=pelton_model, **model)
        return np.array(data_colecole)[self.tindex]

    def constrain_model_vector(self,model_vector):
        resmin = 1e-15
        resmax = 1e15
        taumin = 1e-15
        taumax  = 1e15
        cmin =0
        cmax =1
        model_vector[:self.nlayers] = np.clip(
            model_vector[:self.nlayers], np.log(resmin), np.log(resmax))
        for i in range(self.nlayers):
            resmaxi = model_vector[i]
            model_vector[self.nlayers + i] = np.clip(model_vector[
                    self.nlayers + i],np.log(resmin),resmaxi)
        model_vector[2 * self.nlayers:3 * self.nlayers] = np.clip(
            model_vector[2 * self.nlayers:3 * self.nlayers], np.log(taumin), np.log(taumax))
        model_vector[3 * self.nlayers:4 * self.nlayers] = np.clip(
            model_vector[3 * self.nlayers:4 * self.nlayers], cmin, cmax)
        return model_vector

    def Japprox(self,model_vector, perturbation=0.1, min_perturbation=1e-3):
        delta_m = min_perturbation  # np.max([perturbation*m.mean(), min_perturbation])

        J = []

        for i, entry in enumerate(model_vector):
            mpos = model_vector.copy()
            mpos[i] = entry + delta_m

            mneg = model_vector.copy()
            mneg[i] = entry - delta_m

            pos = self.predicted_data(mpos)
            neg = self.predicted_data(mneg)
            J.append((pos - neg) / (2. * delta_m))

        return np.vstack(J).T

    def get_Wd(self, dobs, ratio=0.01, plateau=1e-4):
        return np.diag(1 / (np.abs(ratio*dobs) + plateau) )

    def steepest_descent_reg(self,dobs, model_init, niter):
        model_vector = model_init
        Wd = self.get_Wd(dobs)
        r =  Wd @ (dobs -self.predicted_data(model_vector))
        f = 0.5*np.dot(r,r)

        error = np.zeros(niter+1)
        error[0] = f
        model_itr = np.zeros((niter+1,model_vector.shape[0]))
        model_itr[0,:] = model_vector

        print(f'Steepest Descent \n initial phid= {f:.3e} ')
        for i in range(niter):
            J = self.Japprox(model_vector)
            r =  Wd @ (dobs - self.predicted_data(model_vector))
            g =   J.T @ Wd.T @ r
            Ag = J @ g
            alpha =  np.mean(Ag * r) / np.mean(Ag * Ag)
            model_vector =  self.constrain_model_vector(model_vector + alpha * g)
            r = Wd @ (self.predicted_data(model_vector) - dobs)
            f = 0.5 * np.dot(r, r)
            if np.linalg.norm(g) < 1e-12:
                break
            error[i+1] = f
            model_itr[i+1, :] = model_vector
            print(f' i= {i:3d}, phid= {f:.3e} ')
    #        error.append([i, mu, norm])
    #    error = np.array(error)
    #    with open('EOSC555_Rep_error_SD2.csv', 'w', newline='') as csvfile:
    #        writer = csv.writer(csvfile)
    #        writer.writerow(['Iteration', 'mu', 'norm'])  # Write header row
    #        writer.writerows(error)  # Write data rows
        return model_vector, error, model_itr

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
    #        error.append([i, mu, norm])
    #    error = np.array(error)
    #    with open('EOSC555_Rep_error_SD2.csv', 'w', newline='') as csvfile:
    #        writer = csv.writer(csvfile)
    #        writer.writerow(['Iteration', 'mu', 'norm'])  # Write header row
    #        writer.writerows(error)  # Write data rows
        return model_vector, error, model_itr

    def steepest_descent_linesearch(self,dobs, model_init, niter,
        alpha0=1, beta=0.5, epsilon=1e-10, mu=1e-4):
        model_old = model_init
        Wd = self.get_Wd(dobs)
        error = np.zeros(niter+1)
        model_itr = np.zeros((niter+1,model_init.shape[0]))
        r_old =  Wd @ (self.predicted_data(model_old) -dobs)
        f_old = 0.5 * np.dot(r_old,r_old)
        error[0] = f_old
        model_itr[0,:] = model_old

        print(f'Steepest Descent, initial phid= {f_old:.3e} ')
        for i in range(niter):
            alpha = alpha0
            J = self.Japprox(model_old)
            r_old =  Wd @ (self.predicted_data(model_old) -dobs)
            g =   J.T @ Wd.T @ r_old
            if np.linalg.norm(g) < epsilon:
                print(f"gradient is small as :{g:.3e} ")
                break
            model_new =  self.constrain_model_vector(model_old - alpha * g)
            r_new = Wd@(dobs - self.predicted_data(model_new))
            directional_derivative = np.dot(g, -g)
            f_old = 0.5 * np.dot(r_old, r_old)
            f_new = 0.5 * np.dot(r_new, r_new)
            while f_new >= f_old + alpha * mu * directional_derivative:
                alpha *= beta
                model_new = self.constrain_model_vector(model_old - alpha * g)
                r_new = Wd@(self.predicted_data(model_new) - dobs)
                f_new = 0.5 * np.dot(r_new, r_new)
                if np.linalg.norm(alpha) < epsilon:
                    break
            model_old = model_new
            error[i+1] = f_new
            model_itr[i+1, :] = model_new
            print(f' i= {i:3d}, alpha= {alpha:.3e}, phid= {f_new:.3e} ')
    #        error.append([i, mu, norm])
    #    error = np.array(error)
    #    with open('EOSC555_Rep_error_SD2.csv', 'w', newline='') as csvfile:
    #        writer = csv.writer(csvfile)
    #        writer.writerow(['Iteration', 'mu', 'norm'])  # Write header row
    #        writer.writerows(error)  # Write data rows
        return model_new,error,model_itr


    def steepest_descent_Reg_LS(self,dobs, model_init, niter, beta,
        alpha0=1, afac=0.5, epsilon=1e-10, mu=1e-4):
        model_old = model_init
        mref = model_init
        Wd = self.get_Wd(dobs)
        error = np.zeros(niter+1)
        model_itr = np.zeros((niter+1,model_init.shape[0]))
        r_old =  Wd @ (self.predicted_data(model_old) -dobs)
        phid = 0.5 * np.dot(r_old,r_old)
        phim = 0.5*  np.dot(model_old-mref,model_old-mref)
        f_old  = phid + beta*phim
        error[0] = f_old
        model_itr[0,:] = model_old

        print(f'Steepest Descent \n Initial phid = {phid:.2e} ,phim = {phim:.2e}, error= {f_old:.2e} ')
        for i in range(niter):
            f_old = phid + beta * phim
            alpha = alpha0
            J = self.Japprox(model_old)
            r_old =  Wd @ (self.predicted_data(model_old) -dobs)
            g =  J.T @ Wd.T @ r_old + beta*(model_old-mref)
            if np.linalg.norm(g) < epsilon:
                print(f"gradient is small as :{g:.3e} ")
                break
            model_new =  self.constrain_model_vector(model_old - alpha * g)
            r_new = Wd@(self.predicted_data(model_new) -dobs)
            directional_derivative = np.dot(g, -g)
            phid = 0.5 * np.dot(r_new, r_new)
            phim = 0.5 * np.dot(model_new - mref, model_new - mref)
            f_new = phid + beta *phim
            while f_new >= f_old + alpha * mu * directional_derivative:
                alpha *= afac
                model_new = self.constrain_model_vector(model_old - alpha * g)
                r_new = Wd@(self.predicted_data(model_new) - dobs)
                phid = 0.5 * np.dot(r_new, r_new)
                phim = 0.5 * np.dot(model_new - mref, model_new - mref)
                f_new = phid +  beta * phim
                if np.linalg.norm(alpha) < epsilon:
                    break
            model_old = model_new
            error[i+1] = f_new
            model_itr[i+1, :] = model_new
            print(f' i= {i:3d}, alpha= {alpha:.2e}, phid = {phid:.2e} ,phim = {phim:.2e}, error= {f_new:.2e} ')
    #        error.append([i, mu, norm])
    #    error = np.array(error)
    #    with open('EOSC555_Rep_error_SD2.csv', 'w', newline='') as csvfile:
    #        writer = csv.writer(csvfile)
    #        writer.writerow(['Iteration', 'mu', 'norm'])  # Write header row
    #        writer.writerows(error)  # Write data rows
        return model_new,error,model_itr

    def get_cc_grid (self, dobs, m_0 ,m_8, m_t,m_c, ngrid=20, mirgin = 0.1):
        m0_grid = np.linspace(np.min(m_0) - mirgin, np.max(m_0) + mirgin, ngrid)
        m08_grid = np.linspace(np.min(m_0 - m_8) - mirgin, np.max(m_0 - m_8) + mirgin, ngrid)
        cc_grid = np.zeros((ngrid, ngrid))
        Wd = self.get_Wd(dobs)

        for j, m0_tmp in enumerate(m0_grid):
            for i, m08_tmp in enumerate(m08_grid):
                model_vector = np.hstack([m0_tmp, m0_tmp - m08_tmp, m_t, m_c])
                r = Wd @ (self.predicted_data(model_vector) - dobs)
                cc_grid[i, j] = 0.5 * np.dot(r, r)
        return m0_grid, m08_grid, cc_grid

# res_air = 2e14
# res_sea = 1/3
# nlayers = 1
# #nlayers = 10
# layer_thicknesses = 5.
# #seabed_depth = 1000.5
# seabed_depth = 1000.5
# #depth = np.hstack([np.r_[0],seabed_depth+layer_thicknesses * np.arange(nlayers)])
# depth = np.array([0,seabed_depth])
# tstrt = 1e-6
# tend = 1e-2
# t = np.logspace(-8,-2, 121)
# tplot = t[(t >= tstrt) & (t <= tend)]
# model_base = {
#     'src':  [1.5,1.5,0,1.5,1000, 1000],
#     'rec': [0,0,1000,0,90],
#     'depth': depth,
#     'freqtime': t ,
#     'signal': 0,
#     'mrec' : True,
#     'verb': 0
# }
#
# EMIP =  EMIP1D(model_base,res_air,res_sea,nlayers)
#
# tstrt = 1e-6
# tend = 1e-2
# tindex = (t >= tstrt) & (t <= tend)
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
# m = (res_0 - res_8) / res_0
#
# print(f'chargeability for obasercation{m}')
# model_obs = np.hstack([m_0, m_8,m_t,m_c])
#
# data_obs = EMIP.predicted_data(model_obs)
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
# niter = 5
# model_SD2 = EMIP.steepest_descent(data_obs, model_init, niter)