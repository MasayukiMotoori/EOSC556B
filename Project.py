import pandas as pd
from scipy import optimize
from scipy.constants import mu_0, epsilon_0

import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import optimize
from scipy.linalg import lu_factor, lu_solve
import empymod


class EMIP1D:
    """
    Class for one dimensional inversion for about Induced Polarization from
    Time-Domain Electromagnetic about Sea Massive Sulfide Exploration.
    Forward modelling: empymod
    https://empymod.emsig.xyz/en/stable/index.html

    Forward simulation

    model        :all parameter which is used for simulation
                   including resistivity of air, sea water.
                   bottom fix is also available

    model vector : parameter used for inversion
                   assuming IP parameter for seafloor hydrothermal deposit
                   
    IP_model: String
        "cole":
            res0: resistivity in low frequency
            res8: resistivity in hig frequency
            tau : time constant
            c   : exponent c

        "pelton":
            res : resistivity
            chg : chargeability
            tau : time constant
            c   : exponent c

    res_air: resistivity of air to fix during inversion

    res_sea: resistivity of sea to fix during inversion

    nlayers: integer, number of layers

    tindex: list of boolen, time index to use for forward modelling and inversion

    btm_fix: indicate if you want to fix parameter for bottom layer
        None: no fix (default)
        True: fix, recommended when creating object function grid

    res_btm: resistivity of bottom layer to fix during inversion
             eligible when btm_fix is 1

    model_base: please refer empymod tutorial
    https://empymod.emsig.xyz/en/stable/api/empymod.model.dipole.html
        'src':  transmitter configuration
        'rec': receiver configuration
        'depth': depth,
        'freqtime': t as time domain
        'signal': signal wave
        'verb': 0
     """

    def __init__(self, IP_model, model_base,
                 res_air, res_sea, nlayers, tindex,
                 btm_fix=None, res_btm=None,
                 resmin=1e-3 , resmax=1e6, chgmin=1e-5, chgmax=0.99,
                 taumin=1e-6, taumax=1e-2, cmin= 0.4, cmax=0.8,
                 Wd = None, Ws=None, Wx=None, Hessian=None, Jacobian=None
                 ):
        self.IP_model = IP_model
        self.model_base = model_base
        self.res_air = res_air
        self.res_sea = res_sea
        self.nlayers = nlayers
        self.tindex = tindex
        self.btm_fix = btm_fix
        self.res_btm = res_btm
        self.resmin = resmin
        self.resmax = resmax
        self.chgmin = chgmin
        self.chgmax = chgmax
        self.taumin = taumin
        self.taumax = taumax
        self.cmin = cmin
        self.cmax = cmax
        self.Wd = Wd
        self.Ws = Ws
        self.Wx = Wx
        self.Hessian = Hessian
        self.Jacobian = Jacobian

    def cole_cole(self, inp, p_dict):
        """Cole and Cole (1941)."""

        # Compute complex conductivity from Cole-Cole
        iotc = np.outer(2j * np.pi * p_dict['freq'], inp['tau']) ** inp['c']
        condH = inp['cond_8'] + (inp['cond_0'] - inp['cond_8']) / (1 + iotc)
        condV = condH / p_dict['aniso'] ** 2

        # Add electric permittivity contribution
        etaH = condH + 1j * p_dict['etaH'].imag
        etaV = condV + 1j * p_dict['etaV'].imag
        PA_plot = 0
        if PA_plot == 1:
            freq = p_dict['freq']
            amplitude = np.abs(1 / condH)[:, 1]
            phase = np.angle(1 / condH)[:, 1]
            #    m_colecole = (inp['cond_8'][2]-inp['cond_0'][2])/inp['cond_8'][2]

            csv_colecole = np.array([freq, amplitude, phase]).T
            with open('EOSC555_Rep_PA_ColeCole.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['freq', 'amplitude', 'phase'])  # Write header row
                writer.writerows(csv_colecole)

        return etaH, etaV

    def pelton_et_al(self, inp, p_dict):
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
            frq_pl = p_dict['freq']
            amp_pl = np.abs(rhoH)[:, 2]
            phs_pl = np.angle(rhoH)[:, 2]
            fig, ax1 = plt.subplots(figsize=(8, 5))

            # Plotting magnitude on the left y-axis (log scale)
            ax1.set_xscale('log')
            ax1.set_ylabel('Amplitude')
            ax1.set_xlabel('Frequency')
            ax1.semilogx(frq_pl, amp_pl, 'r',label='Amplitude')  # Blue line for magnitude
            ax1.set_ylim(bottom=0)  # Set the lower limit of the y-axis to 0

            # Enable grid only for x-axis
            ax1.xaxis.grid(True)
            ax1.yaxis.grid(False)  # Disable grid for y-axis

            # Creating a second y-axis for phase on the right
            ax2 = ax1.twinx()
            ax2.set_ylabel('Phase (radians)')
            ax2.semilogx(frq_pl, phs_pl,  'b-',label='phase')  # Red line for phase
            ax2.invert_yaxis()

            # Combine legends from both axes
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines = lines1 + lines2
            labels = labels1 + labels2
            ax1.legend(lines, labels, loc='best')

            plt.title('Phase Amplitude Plot-Pelton model')
            plt.savefig('Pelton.png', dpi=300)
            plt.show()
            # csv_colecole = np.array([freq, amplitude, phase]).T
            # with open('EOSC555_Rep_PA_Pelton.csv', 'w', newline='') as csvfile:
            #     writer = csv.writer(csvfile)
            #     writer.writerow(['freq', 'amplitude', 'phase'])  # Write header row
            #     writer.writerows(csv_colecole)

        return etaH, etaV

    def get_ip_model(self, model_vector):
        if self.IP_model == "cole":
            if self.btm_fix == True:
                res_0 = np.hstack([[self.res_air, self.res_sea],
                                   np.exp(model_vector[:self.nlayers]), [self.res_btm]])
                res_8 = np.hstack([[self.res_air, self.res_sea],
                                   np.exp(model_vector[self.nlayers:2 * self.nlayers]), [self.res_btm]])
                tau = np.hstack([[1e-3, 1e-3],
                                 np.exp(model_vector[2 * self.nlayers:3 * self.nlayers]), 1e-3])
                c = np.hstack([[0.5, 0.5], model_vector[3 * self.nlayers:4 * self.nlayers], [0.5]])
                cole_model = {'res': res_0, 'cond_0': 1 / res_0, 'cond_8': 1 / res_8,
                              'tau': tau, 'c': c, 'func_eta': self.cole_cole}
                return cole_model
            else:
                res_0 = np.hstack([[self.res_air, self.res_sea],
                                   np.exp(model_vector[:self.nlayers])])
                res_8 = np.hstack([[self.res_air, self.res_sea],
                                   np.exp(model_vector[self.nlayers:2 * self.nlayers])])
                tau = np.hstack([[1e-3, 1e-3],
                                 np.exp(model_vector[2 * self.nlayers:3 * self.nlayers])])
                c = np.hstack([[0.5, 0.5], model_vector[3 * self.nlayers:4 * self.nlayers]])
                cole_model = {'res': res_0, 'cond_0': 1 / res_0, 'cond_8': 1 / res_8,
                              'tau': tau, 'c': c, 'func_eta': self.cole_cole}
                return cole_model

        if self.IP_model == "pelton":
            if self.btm_fix == True:
                res = np.hstack([[self.res_air, self.res_sea],
                                 np.exp(model_vector[:self.nlayers]), [self.res_btm]])
                m = np.hstack([[0., 0.], model_vector[self.nlayers:2 * self.nlayers], [0.]])
#                 m = np.hstack([[0., 0.],
#                                np.exp(model_vector[self.nlayers:2 * self.nlayers]), [0.]])
                tau = np.hstack([[1e-3, 1e-3],
                                 np.exp(model_vector[2 * self.nlayers:3 * self.nlayers]), 1e-3])
                c = np.hstack([[0.5, 0.5], model_vector[3 * self.nlayers:4 * self.nlayers], [0.5]])
                pelton_model = {'res': res, 'rho_0': res, 'm': m,
                                'tau': tau, 'c': c, 'func_eta': self.pelton_et_al}
                return pelton_model
            else:
                res = np.hstack([[self.res_air, self.res_sea],
                                 np.exp(model_vector[:self.nlayers])])
                m = np.hstack([[0., 0.], model_vector[self.nlayers:2 * self.nlayers]])
                # m = np.hstack([[0., 0.], np.exp(model_vector[self.nlayers:2 * self.nlayers])])
                tau = np.hstack([[1e-3, 1e-3],
                                 np.exp(model_vector[2 * self.nlayers:3 * self.nlayers])])
                c = np.hstack([[0.5, 0.5], model_vector[3 * self.nlayers:4 * self.nlayers]])
                pelton_model = {'res': res, 'rho_0': res, 'm': m,
                                'tau': tau, 'c': c, 'func_eta': self.pelton_et_al}
                return pelton_model

    def plot_model(self, model, ax, color, label="model", linewidth=1):
        depth = self.model_base["depth"]
        depth_plot = np.vstack([depth, depth]).flatten(order="F")[1:]
        depth_plot = np.hstack([depth_plot, depth_plot[-1] * 1.5])
        model_plot = np.vstack([model, model]).flatten(order="F")[2:]
        return ax.plot(model_plot, depth_plot, color, label=label,  linewidth=linewidth)

    def predicted_data(self, model_vector):

        ip_model = self.get_ip_model(model_vector)
        data = empymod.bipole(res=ip_model, **self.model_base)
        return np.array(data)[self.tindex]

    def constrain_model_vector(self, model_vector):
        "Parametert Projection based on provided bound"

        model_vector[:self.nlayers] = np.clip(
            model_vector[:self.nlayers], np.log(self.resmin), np.log(self.resmax))
        if self.IP_model == "cole":
            for i in range(self.nlayers):
                resmaxi = model_vector[i]
                model_vector[self.nlayers + i] = np.clip(
                model_vector[self.nlayers + i], np.log(self.resmin), resmaxi)
        else:
            model_vector[self.nlayers:2 * self.nlayers] = np.clip(
                model_vector[self.nlayers:2 * self.nlayers], self.chgmin, self.chgmax)
            # model_vector[self.nlayers:2 * self.nlayers] = np.clip(
            #     model_vector[self.nlayers:2 * self.nlayers], np.log(self.chgmin), np.log(self.chgmax))
        model_vector[2 * self.nlayers:3 * self.nlayers] = np.clip(
            model_vector[2 * self.nlayers:3 * self.nlayers], np.log(self.taumin), np.log(self.taumax))
        model_vector[3 * self.nlayers:4 * self.nlayers] = np.clip(
            model_vector[3 * self.nlayers:4 * self.nlayers], self.cmin, self.cmax)
        return model_vector


    def projection_halfspace(self,a,x,b):
        "project vector to half space {x | <a,x> <=b "
        if np.all(np.dot(a,x) <= b):
            return x
        else:
            return x + a*( (b-np.dot(a,x))/np.dot(a,a) )

    def proj_c(self,mvec_tmp,maxitr = 100, tol = 1e-6):
        "Project model vector to convex set defined by bound information"
        mvec = mvec_tmp.copy()
        nlayers = self.nlayers
        a_r0  = np.r_[ 1., 0.]
        # a_r8  = np.r_[ 0., 1.]
        a_r08 = np.r_[-1., 1.]
        a = np.r_[1]

        if self.IP_model == "cole" :
            for j in range(nlayers):
                r0_tmp = mvec[j]
                r8_tmp = mvec[j + nlayers]
                r08_tmp = np.r_[r0_tmp, r8_tmp]
                r08_prj = r08_tmp
                for i in range(maxitr):
                    r08_prj = self.projection_halfspace(  a_r0, r08_prj,  np.log(self.resmax))
                    r08_prj = self.projection_halfspace( -a_r0, r08_prj, -np.log(self.resmin))
                    # r08_prj = self.projection_halfspace( a_r8, r08_prj,   np.log(self.resmax))
                    # r08_prj = self.projection_halfspace(-a_r8, r08_prj, -np.log(self.resmin))
                    r08_prj = self.projection_halfspace(-a_r08, r08_prj, -np.log(1-self.chgmax))
                    r08_prj = self.projection_halfspace( a_r08, r08_prj,  np.log(1-self.chgmin))
                    if np.linalg.norm(r08_prj - r08_tmp) <= tol:
                        break
                    r08_tmp = r08_prj
                mvec[j          ] = r08_prj[0]
                mvec[j + nlayers] = r08_prj[1]

        else:
            for j in range(nlayers):
                r_prj = mvec[j]
                m_prj = mvec[j+nlayers]
                r_prj = self.projection_halfspace( a, r_prj, np.log(self.resmax))
                r_prj = self.projection_halfspace(-a, r_prj,-np.log(self.resmin))
                m_prj = self.projection_halfspace( a, m_prj, self.chgmax)
                m_prj = self.projection_halfspace(-a, m_prj,-self.chgmin)
                mvec[j        ] = r_prj
                mvec[j+nlayers] = m_prj

        for j in range(nlayers):
            t_prj = mvec[j + 2*nlayers]
            c_prj = mvec[j + 3*nlayers]
            t_prj = self.projection_halfspace( a, t_prj,  np.log(self.taumax))
            t_prj = self.projection_halfspace(-a, t_prj, -np.log(self.taumin))
            c_prj = self.projection_halfspace( a, c_prj,  self.cmax)
            c_prj = self.projection_halfspace(-a, c_prj, -self.cmin)
            mvec[j + 2*nlayers] = t_prj
            mvec[j + 3*nlayers] = c_prj
        return mvec


    def Japprox(self, model_vector, perturbation=0.1, min_perturbation=1e-3):
        delta_m = min_perturbation  # np.max([perturbation*m.mean(), min_perturbation])
#        delta_m = perturbation  # np.max([perturbation*m.mean(), min_perturbation])
        J = []

        for i, entry in enumerate(model_vector):
            mpos = model_vector.copy()
            mpos[i] = entry + delta_m

            mneg = model_vector.copy()
            mneg[i] = entry - delta_m

            mpos = self.constrain_model_vector(mpos)
            mneg = self.constrain_model_vector(mneg)


            pos = self.predicted_data(self.constrain_model_vector(mpos))
            neg = self.predicted_data(self.constrain_model_vector(mneg))
            J.append((pos - neg) / (2. * delta_m))

        return np.vstack(J).T

    def get_Wd(self, dobs, dp=1, ratio=0.01, plateau=1e-5):
        """
        Calculate and return a diagonal matrix Wd based on a standard deviation std.

        The standard deviation std is calculated as follows: (abs(dobs * ratio) ^ dp) + plateau.
        The diagonal matrix Wd is then calculated as the inverse of the standard deviation std.

        Parameters
        ----------
        dobs : ndarray
            The observed data.
        dp : int, optional
            The power to which the absolute value of dobs * ratio is raised (default is 1).
        ratio : float, optional
            A constant used in the calculation of the standard deviation (default is 0.01).
        plateau : float, optional
            A small constant added to avoid division by zero (default is 1e-5).

        Returns
        -------
        ndarray
            A diagonal matrix with the elements of 1 / std on the main diagonal.
        """
        std = np.abs(dobs * ratio) ** dp + plateau
        return np.diag(1 / std)

    def set_Wd(self, dobs, dp=1, ratio=0.01, plateau=1e-5 ):
        """
        Calculate a diagonal matrix Wd based on a standard deviation std and set it as an attribute of the class instance.

        The standard deviation std is calculated as follows: (abs(dobs * ratio) ^ dp) + plateau.
        The diagonal matrix Wd is then calculated as the inverse of the standard deviation std.

        Parameters
        ----------
        dobs : ndarray
            The observed data.
        dp : int, optional
            The power to which the absolute value of dobs * ratio is raised (default is 1).
        ratio : float, optional
            A constant used in the calculation of the standard deviation (default is 0.01).
        plateau : float, optional
            A small constant added to avoid division by zero (default is 1e-5).

        Returns
        -------
        None
        """
        std = np.abs(dobs * ratio) ** dp + plateau
        self.Wd = np.diag(1 / std)
        return

    def get_Ws(self):
        nx = 4*self.nlayers
        return np.diag(np.ones(nx))

    def set_Ws(self):
        nx = 4*self.nlayers
        self.Ws = np.diag(np.ones(nx))
        return

    def get_Wx(self):
        """
        Calculate a matrix Wx that represents the smoothness constraint for a model with multiple layers 
        and set it as an attribute of the class instance.

        The method first initializes Wx as a zero matrix of size (4 * nx, 4 * ny), where nx is the number of layers minus one 
        and ny is the number of layers.

        If the number of layers is one, the method prints a message "No smoothness for one layer model", 
        sets Wx as a zero matrix of size (4,4), and returns.

        For models with more than one layer, the method constructs Wx such that it has -1 on the main diagonal 
        and 1 on the diagonal above it, for each block of size nx by ny in Wx.

        Parameters
        ----------
        None

        Returns
        -------
        ndarray
            A matrix that represents the smoothness constraint for a model with multiple layers.
        """

        nx = self.nlayers - 1
        ny = self.nlayers
        Wx = np.zeros((4 * nx, 4 * ny))
        if self.nlayers == 1:
            print("No smoothness for one layer model")
            Wx = np.zeros((4,4))
            return Wx
        for i in range(4):
            Wx[i * nx:(i + 1) * nx, i * ny:(i + 1) * ny - 1] = -np.diag(np.ones(nx))
            Wx[i * nx:(i + 1) * nx, i * ny + 1:(i + 1) * ny] += np.diag(np.ones(nx))
        return Wx

    def set_Wx(self):
        """
        Calculate a matrix Wx that represents the smoothness constraint for a model with multiple layers 
        and set it as an attribute of the class instance.

        The method first initializes Wx as a zero matrix of size (4 * nx, 4 * ny), where nx is the number of layers minus one 
        and ny is the number of layers.

        If the number of layers is one, the method prints a message "No smoothness for one layer model", 
        sets Wx as a zero matrix of size (4,4), and returns.

        For models with more than one layer, the method constructs Wx such that it has -1 on the main diagonal 
        and 1 on the diagonal above it, for each block of size nx by ny in Wx.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        nx = self.nlayers - 1
        ny = self.nlayers
        Wx = np.zeros((4 * nx, 4 * ny))
        if self.nlayers == 1:
            print("No smoothness for one layer model")
            Wx = np.zeros((4,4))
            self.Wx = Wx
            return
        for i in range(4):
            Wx[i * nx:(i + 1) * nx, i * ny:(i + 1) * ny - 1] = -np.diag(np.ones(nx))
            Wx[i * nx:(i + 1) * nx, i * ny + 1:(i + 1) * ny] += np.diag(np.ones(nx))
        self.Wx = Wx
        return

    def get_Wxx(self):

        e = np.ones(self.nlayers*4)

        p1 = np.ones(self.nlayers)
        p1[0] = 2
        p1[-1] = 0
        eup = np.tile(p1, 4)

        p2 = np.ones(self.nlayers)
        p2[0] = 0
        p2[-1] = 2
        edwn = np.tile(p2, 4)
        Wxx = np.diag(-2 * e) + np.diag(eup[:-1], 1) + np.diag(edwn[1:], -1)

        return Wxx

    def steepest_descent(self, dobs, model_init, niter):
        model_vector = model_init
        r = dobs - self.predicted_data(model_vector)
        f = 0.5 * np.dot(r, r)

        error = np.zeros(niter + 1)
        error[0] = f
        model_itr = np.zeros((niter + 1, model_vector.shape[0]))
        model_itr[0, :] = model_vector

        print(f'Steepest Descent \n initial phid= {f:.3e} ')
        for i in range(niter):
            J = self.Japprox(model_vector)
            r = dobs - self.predicted_data(model_vector)
            dm = J.T @ r
            g = np.dot(J.T, r)
            Ag = J @ g
            alpha = np.mean(Ag * r) / np.mean(Ag * Ag)
            model_vector = self.constrain_model_vector(model_vector + alpha * dm)
            r = self.predicted_data(model_vector) - dobs
            f = 0.5 * np.dot(r, r)
            if np.linalg.norm(dm) < 1e-12:
                break
            error[i + 1] = f
            model_itr[i + 1, :] = model_vector
            print(f' i= {i:3d}, phid= {f:.3e} ')
        return model_vector, error, model_itr


    def Gradient_Descent(self, dobs, mvec_init, niter, beta, alphas, alphax,
            s0=1, sfac=0.5, stol=1e-6, gtol=1e-3, mu=1e-4, ELS=True, BLS=True ):
        """
        Perform the Gradient Descent algorithm for optimization.

        Parameters
        ----------
        dobs : ndarray
            The observed data.
        mvec_init : ndarray
            The initial model vector.
        niter : int
            The number of iterations to perform.
        beta : float
            The beta parameter for the algorithm.
        alphas : float
            The alpha_s parameter for the algorithm.
        alphax : float
            The alpha_x parameter for the algorithm.
        s0 : float, optional
            The initial step size (default is 1).
        sfac : float, optional
            The step size reduction factor (default is 0.5).
        stol : float, optional
            The step size tolerance (default is 1e-6).
        gtol : float
            The stopping criteria for the norm of the gradient.
        mu : float, optional
            The mu parameter for the algorithm (default is 1e-4).
        ELS : bool, optional
            Whether to use exact line search (default is True).
        BLS : bool, optional
            Whether to use backtracking line search (default is True).

        Returns
        -------
        mvec_new : ndarray
            The optimized model vector.
        error_prg : ndarray
            The progress of the error.
        mvec_prg : ndarray
            The progress of the model vector.

        """
        Wd = self.Wd
        Ws = self.Ws
        Wx = self.Wx

        mvec_old = mvec_init
        mvec_new = None
        mref = mvec_init
        error_prg = np.zeros(niter + 1)
        mvec_prg = np.zeros((niter + 1, mvec_init.shape[0]))
        rd = Wd @ (self.predicted_data(mvec_old) - dobs)
        phid = 0.5 * np.dot(rd, rd)
        rms = 0.5 * np.dot(Ws@(mvec_old - mref), Ws@(mvec_old - mref))
        rmx = 0.5 * np.dot(Wx @ mvec_old, Wx @ mvec_old)
        phim = alphas * rms + alphax * rmx
        f_old = phid + beta * phim
        k = 0
        error_prg[0] = f_old
        mvec_prg[0, :] = mvec_old
        print(f'Gradient Descent \n Initial phid = {phid:.2e} ,phim = {phim:.2e}, error= {f_old:.2e} ')
        for i in range(niter):
            # Calculate J:Jacobian and g:gradient
            J = self.Japprox(mvec_old)
            g = J.T @ Wd.T @ rd + beta * (alphas * Ws.T @ Ws @ (mvec_old - mref)
                                          + alphax * Wx.T @ Wx @ mvec_old)
            # Exact line search
            if ELS:
                t = np.dot(g,g)/np.dot(Wd@J@g,Wd@J@g)
#                t = (g.T@g)/(g.T@J.T@J@g)
            else:
                t = 1.

            # End inversion if gradient is smaller than tolerance
            g_norm = np.linalg.norm(g, ord=2)
            if g_norm < gtol:
                print(f"Inversion complete since norm of gradient is small as :{g_norm :.3e} ")
                break

            # Line search method Armijo using directional derivative
            s = s0
            dm = t*g
            directional_derivative = np.dot(g, -dm)

            mvec_new = self.proj_c(mvec_old - s * dm)
            rd = Wd @ (self.predicted_data(mvec_new) - dobs)
            phid = 0.5 * np.dot(rd, rd)
            rms = 0.5 * np.dot(Ws @ (mvec_new - mref), Ws @ (mvec_new - mref))
            rmx = 0.5 * np.dot(Wx @ mvec_new, Wx @ mvec_new)
            phim = alphas * rms + alphax * rmx
            f_new = phid + beta * phim
            if BLS:
                while f_new >= f_old + s * mu * directional_derivative:
                    s *= sfac
                    mvec_new = self.proj_c(mvec_old - s * dm)
                    rd = Wd @ (self.predicted_data(mvec_new) - dobs)
                    phid = 0.5 * np.dot(rd, rd)
                    rms = 0.5 * np.dot(Ws @ (mvec_new - mref), Ws @ (mvec_new - mref))
                    rmx = 0.5 * np.dot(Wx @ mvec_new, Wx @ mvec_new)
                    phim = alphas * rms + alphax * rmx
                    f_new = phid + beta * phim
                    if np.linalg.norm(s) < stol:
                        break
            mvec_old = mvec_new
            mvec_prg[i + 1, :] = mvec_new
            f_old = f_new
            error_prg[i + 1] = f_new
            k = i + 1
            print(f'{k:3}, s:{s:.2e}, gradient:{g_norm:.2e}, phid:{phid:.2e}, phim:{phim:.2e}, f:{f_new:.2e} ')
        # filter model prog data
        mvec_prg = mvec_prg[:k]
        error_prg = error_prg[:k]
        # Save Jacobian
        self.Jacobian = J
        return mvec_new, error_prg, mvec_prg

    def GaussNewton_smooth(self, dobs, mvec_init, niter,
                           beta, alphas, alphax,
                           s0=1, sfac=0.5, stol=1e-6, gtol=1e-3, mu=1e-4):
        """
        Implements the Gauss-Newton method with smooth regularization for solving an inverse problem.

        Parameters
        ----------
        dobs : ndarray
            The observed data.
        mvec_init : ndarray
            The initial model vector.
        niter : int
            The maximum number of iterations.
        beta : float
            The beta value for the model part.
        alphas : float
            The alpha values for smallness.
        alphax : float
            The alpha values for smoothness.
        s0 : float
            The initial step size for the Armijo line search.
        sfac : float
            The step size reduction factor for the Armijo line search.
        stol : float
            The step size tolerance for the Armijo line search.
        gtol : float
            The stopping criteria for the norm of the gradient.
        mu : float
            The mu parameter for the Armijo line search.

        Returns
        -------
        mvec : ndarray
            The final model vector.
        eprogress : list
            The progress of the error.
        mprogress : list
            The progress of the model vector.
        """

        Wd = self.Wd
        Ws = self.Ws
        Wx = self.Wx
        mvec_old = mvec_init
        # applay initial mvec for reference mode
        mref = mvec_init
        # get noise part
        # Wd = self.get_Wd(dobs)
        # Initialize object function
        rd = Wd @ (self.predicted_data(mvec_old) - dobs)
        phid = 0.5 * np.dot(rd, rd)
        # Ws = self.get_Ws()
        # Wx = self.get_Wx()
        rms = 0.5 * np.dot(mvec_old - mref, mvec_old - mref)
        rmx = 0.5 * np.dot(Wx @ mvec_old, Wx @ mvec_old)
        phim = alphas * rms + alphax * rmx
        f_old = phid + beta * phim
        # Prepare array for storing error and model in progress
        error_prg = np.zeros(niter + 1)
        mvec_prg = np.zeros((niter + 1, mvec_init.shape[0]))
        error_prg[0] = f_old
        mvec_prg[0, :] = mvec_old

        print(f'Gauss-Newton \n Initial phid = {phid:.2e} ,phim = {phim:.2e}, error= {f_old:.2e} ')
        for i in range(niter):

            # Jacobian
            J = self.Japprox(mvec_old)

            # gradient
            g = J.T @ Wd.T @ rd + beta * (alphas * Ws.T @ Ws @ (mvec_old - mref)
                                          + alphax * Wx.T @ Wx @ mvec_old)

            # Hessian approximation

            H = J.T @ Wd.T @ Wd @ J + beta * (alphas * Ws.T @ Ws + alphax * Wx.T @ Wx)

            # model step
            dm = np.linalg.solve(H, g)

            # End inversion if gradient is smaller than tolerance
            g_norm = np.linalg.norm(g, ord=2)
            if g_norm < gtol:
                print(f"Inversion complete since norm of gradient is small as :{g_norm :.3e} ")
                break

            # update object function
            s = s0
            mvec_new = self.proj_c(mvec_old - s * dm)
            rd = Wd @ (self.predicted_data(mvec_new) - dobs)
            phid = 0.5 * np.dot(rd, rd)
            rms = 0.5 * np.dot(mvec_new - mref, mvec_new - mref)
            rmx = 0.5 * np.dot(Wx @ mvec_new, Wx @ mvec_new)
            phim = alphas * rms + alphax * rmx
            f_new = phid + beta * phim

            # Backtracking method using directional derivative Amijo
            directional_derivative = np.dot(g, -dm)
            while f_new >= f_old + s * mu * directional_derivative:
                # backtracking
                s *= sfac
                # update object function
                mvec_new = self.proj_c(mvec_old - s * dm)
                rd = Wd @ (self.predicted_data(mvec_new) - dobs)
                phid = 0.5 * np.dot(rd, rd)
                rms = 0.5 * np.dot(Ws @ (mvec_new - mref), Ws @ (mvec_new - mref))
                rmx = 0.5 * np.dot(Wx @ mvec_new, Wx @ mvec_new)
                phim = alphas * rms + alphax * rmx
                f_new = phid + beta * phim
                # Stopping criteria for backtrackinng
                if s < stol:
                    break

            # Update model
            mvec_old = mvec_new
            mvec_prg[i + 1, :] = mvec_new
            f_old = f_new
            error_prg[i + 1] = f_new
            k = i + 1
            print(f'{k:3}, step:{s:.2e}, gradient:{g_norm:.2e}, phid:{phid:.2e}, phim:{phim:.2e}, f:{f_new:.2e} ')
        # clip progress of model and error in inversion
        error_prg = error_prg[:k]
        mvec_prg = mvec_prg[:k]
        # Save Jacobian and Hessian
        self.Jacobian = J
        self.Hessian = H

        return mvec_new, error_prg, mvec_prg

    def get_r08_grid(self,  mr0lim, mr8lim, m_t, m_c,
        dobs, mref,
        beta,  alphas, alphax, ngrid=20, mirgin=0.1, null_value=-1):
        # return grid of object function with respect to resistivity in high and low frequency
        # assuming IP model is cole model tau and c are fixed value.
        if self.IP_model == "pelton":
            print("use rm_grid for pelton model")
            return
        Wd = self.Wd
        Ws = self.Ws
        Wx = self.Wx
        mr0_grid0 = (1+mirgin)*(np.min(mr0lim)) - mirgin* (np.max(mr0lim))
        mr0_grid1 = (1+mirgin)*(np.max(mr0lim)) - mirgin* (np.min(mr0lim))
        mr0_grid = np.linspace(mr0_grid0, mr0_grid1, ngrid)
        mr8_grid0 = (1+mirgin)*(np.min(mr8lim)) - mirgin* (np.max(mr8lim))
        mr8_grid1 = (1+mirgin)*(np.max(mr8lim)) - mirgin* (np.min(mr8lim))
        mr8_grid = np.linspace(mr8_grid0, mr8_grid1, ngrid)
        r08_grid = np.zeros((ngrid, ngrid))

        for j, mr0_tmp in enumerate(mr0_grid):
            for i, mr8_tmp in enumerate(mr8_grid):
                if -mr0_tmp + mr8_tmp >= np.log(1-self.chgmax) and -mr0_tmp + mr8_tmp <= np.log(1-self.chgmin) :
                    mvec = np.hstack([mr0_tmp, mr8_tmp, m_t, m_c])
                    r = Wd @ (self.predicted_data(mvec) - dobs)
                    phid = 0.5 * np.dot(r, r)
                    rms = 0.5 * np.dot(Ws @ (mvec - mref), Ws @ (mvec - mref))
                    rmx = 0.5 * np.dot(Wx @ mvec, Wx @ mvec)
                    phim = alphas * rms + alphax * rmx
                    r08_grid[i, j] = phid + beta * phim
                else:
                    r08_grid[i, j] = null_value

        r08_grid = np.ma.masked_values(r08_grid, null_value)
        return mr0_grid, mr8_grid, r08_grid

    def get_rm_grid(self,  mrlim, mmlim, m_t, m_c,
        dobs, mref,
        beta, alphas, alphax, ngrid=20, mirgin=0.1):
        # return grid of object function with respect to resistivity in high and low frequency
        # assuming IP model is pelton model tau and c are fixed value.
        if self.IP_model == "cole":
            print("use r08_grid for cole model")
            return
        Wd = self.Wd
        Ws = self.Ws
        Wx = self.Wx
        mr_grid0 = (1+mirgin)*(np.min(mrlim)) - mirgin* (np.max(mrlim))
        mr_grid1 = (1+mirgin)*(np.max(mrlim)) - mirgin* (np.min(mrlim))
        mr_grid = np.linspace(mr_grid0, mr_grid1, ngrid)
        mm_grid0 = (1+mirgin)*(np.min(mmlim)) - mirgin* (np.max(mmlim))
        mm_grid1 = (1+mirgin)*(np.max(mmlim)) - mirgin* (np.min(mmlim))
        mm_grid = np.linspace(mm_grid0, mm_grid1, ngrid)
        rm_grid = np.zeros((ngrid, ngrid))

        for j, mr_tmp in enumerate(mr_grid):
            for i, mm_tmp in enumerate(mm_grid):
                mvec = np.hstack([mr_tmp, mm_tmp, m_t, m_c])
                r = Wd @ (self.predicted_data(mvec) - dobs)
                phid = 0.5 * np.dot(r, r)
                rms = 0.5 * np.dot(Ws @ (mvec - mref), Ws @ (mvec - mref))
                rmx = 0.5 * np.dot(Wx @ mvec, Wx @ mvec)
                phim = alphas * rms + alphax * rmx
                rm_grid[i, j] = phid + beta * phim

        return rm_grid, mm_grid, mr_grid


    def get_tc_grid(self, m_r, m_m, mtlim, mclim,
        dobs, mref,
        beta, alphas, alphax, ngrid=20, mirgin=0.1, null_value=-1):
        # return grid of object function with respect to resistivity in high and low frequency
        # assuming IP model is cole model tau and c are fixed value.
        Wd = self.Wd
        Ws = self.Ws
        Wx = self.Wx
        mt_grid0 = (1+mirgin)*(np.min(mtlim)) - mirgin* (np.max(mtlim))
        mt_grid1 = (1+mirgin)*(np.max(mtlim)) - mirgin* (np.min(mtlim))
        mt_grid = np.linspace(mt_grid0, mt_grid1, ngrid)
        mc_grid0 = (1+mirgin)*(np.min(mclim)) - mirgin* (np.max(mclim))
        mc_grid1 = (1+mirgin)*(np.max(mclim)) - mirgin* (np.min(mclim))
        mc_grid = np.linspace(mc_grid0, mc_grid1, ngrid)
        tc_grid = np.zeros((ngrid, ngrid))

        for j, mt_tmp in enumerate(mt_grid):
            for i, mc_tmp in enumerate(mc_grid):
                mvec = np.hstack([m_r, m_m, mt_tmp, mc_tmp])
                r = Wd @ (self.predicted_data(mvec) - dobs)
                phid = 0.5 * np.dot(r, r)
                rms = 0.5 * np.dot(Ws @ (mvec - mref), Ws @ (mvec - mref))
                rmx = 0.5 * np.dot(Wx @ mvec, Wx @ mvec)
                phim = alphas * rms + alphax * rmx
                tc_grid[i, j] = phid + beta * phim
        return tc_grid, mt_grid, mc_grid


    def plot_IP_par(self,mvec,color="orange",  label="",  linewidth=1.0,ax=None):
        """"
        Return four plots about four IP parameters on given ax.
        mvec: model vector
        label, color: will be reflected on 
        color: 
        ax: Assuming four element in axis 0.
        """""
        if ax == None:
            fig, ax = plt.subplots(2, 2, figsize=(12, 8))

        # convert model vector to model
        model = self.get_ip_model(mvec)

        # plot_model_m(model_base["depth"], model_ip["res"], ax[0], "resistivity","k")
        self.plot_model(model["res"], ax[0], color, label=label, linewidth=linewidth)
        if self.IP_model == "cole":
            self.plot_model(1 - model["cond_0"] / model["cond_8"], ax[1], color, label=label, linewidth=linewidth)
        else:
            self.plot_model(model["m"], ax[1],  color, label=label, linewidth=linewidth)

        self.plot_model(model["tau"], ax[2],  color, label=label, linewidth=linewidth)

        self.plot_model(model["c"]  , ax[3],  color, label=label, linewidth=linewidth)

        ax[0].set_title("model_resistivity(ohm-m)")
        ax[1].set_title("model_changeability")
        ax[2].set_title("model_time_constant(sec)")
        ax[3].set_title("model_exponent_c")

        return ax


class psuedolog():
    """
    A class used to create pseudolog plots.
    ...

    Attributes
    ----------
    posmax : float
        The maximum positive value for the pseudolog.
    negmax : float
        The maximum negative value for the pseudolog.
    a : float
        The value used to scale the pseudolog.
    b : float
        The value used to shift the p
        seudolog.

    Methods
    -------
    pl_plot(x, yinp, ax=None, color="orange", label="pl_plot"):
        Converts the input yinp into a pseudolog and plots it against x on a semilogx plot.
    pl_scatter(x, yinp, ax=None, marker="o",s=5,color="orange", label="pl_plot"):
        Converts the input yinp into a pseudolog and creates a scatter plot against x on a semilogx plot.
    pl_axes(ax=None):
        Adjusts the y-axis of the plot to be suitable for pseudolog data.
    """
    def __init__(self, posmax, negmax, a, b):
        """
        Constructs all the necessary attributes for the psuedolog object.

        Parameters
        ----------
            posmax : float
                The maximum positive value for the pseudolog.
            negmax : float
                The maximum negative value for the pseudolog.
            a : float
                The value used to scale the pseudolog.
            b : float
                The value used to shift the pseudolog.
        """
        self.posmax = posmax
        self.negmax = negmax
        self.a = a
        self.b = b

    def pl_plot(self, x, yinp, ax=None, color="orange", label="pl_plot"):
        """
        Converts the input yinp into a pseudolog and plots it against x on a semilogx plot.

        Parameters
        ----------
            x : ndarray
                The x values for the plot.
            yinp : ndarray
                The y values to be converted into a pseudolog.
            ax : AxesSubplot, optional
                The axes upon which to plot. If None, a new plot is created.
            color : str, optional
                The color of the plot.
            label : str, optional
                The label for the plot.

        Returns
        -------
            ax : AxesSubplot
                The axes with the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        a = self.a
        b = self.b

        ny = yinp.shape[0]
        y = np.zeros_like(yinp)

        # Converting Psuedolog
        for i in range(ny):
            if abs(yinp[i]) >= a:
                y[i] = np.sign(yinp[i]) * (np.log10(abs(yinp[i] / a)) + b)
            else:
                y[i] = yinp[i] / a * b
        ax.semilogx(x, y, color=color, label=label)
        return ax

    def pl_scatter(self, x, yinp, ax=None, marker="o",s=5,color="orange", label="pl_plot"):
        """
        Converts the input yinp into a pseudolog and creates a scatter plot against x on a semilogx plot.

        Parameters
        ----------
            x : ndarray
                The x values for the scatter plot.
            yinp : ndarray
                The y values to be converted into a pseudolog.
            ax : AxesSubplot, optional
                The axes upon which to plot. If None, a new plot is created.
            marker : str, optional
                The marker style for the scatter plot.
            s : int, optional
                The size of the markers for the scatter plot.
            color : str, optional
                The color of the scatter plot.
            label : str, optional
                The label for the scatter plot.

        Returns
        -------
            ax : AxesSubplot
                The axes with the scatter plot.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        a = self.a
        b = self.b

        ny = yinp.shape[0]
        y = np.zeros_like(yinp)

        # Converting Psuedolog
        for i in range(ny):
            if abs(yinp[i]) >= a:
                y[i] = np.sign(yinp[i]) * (np.log10(abs(yinp[i] / a)) + b)
            else:
                y[i] = yinp[i] / a * b
        ax.scatter(x, y, marker=marker,s=s,color=color, label=label)
        ax.set_xscale('log')  # Set x-axis scale to logarithmic
        return ax


    def pl_axes(self,ax=None):
        """
        Adjusts the y-axis of the plot to be suitable for pseudolog data.

        Parameters
        ----------
            ax : AxesSubplot, optional
                The axes to be adjusted. If None, a new plot is created.

        Returns
        -------
            ax : AxesSubplot
                The axes with the adjusted y-axis.
        """
        posmax = self.posmax
        negmax = self.negmax
        a = self.a
        b = self.b

        if negmax > 0:
            if negmax <= a:
                negticks = -b
                neglabels = -a
            else:
                n_negtick= int(np.ceil(np.log10(negmax/a)+1))
                negticks = -b - np.arange(n_negtick-1,-1,-1)
                negmaxlogint = int( np.log10(negmax))
                neglabels = -np.logspace(negmaxlogint,np.log10(a),  n_negtick)
        else:
            negticks  = []
            neglabels = []

        posmaxlogint = int( np.ceil(np.log10(posmax)))
        n_postick= int(np.ceil(np.log10(posmax/a)+1))
        posticks = b + np.arange(n_postick)
        poslabels = np.logspace(np.log10(a), posmaxlogint, n_postick)
        ticks  = np.hstack(( negticks, [0], posticks))
        labels = np.hstack((neglabels, [0], poslabels))
        # Plot figure
        ax.set_ylim([min(ticks), max(ticks)])
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        ax.grid(True)

        return ax

