import torch
import torch.nn.functional as F
import torch.nn as nn
from scipy.stats import qmc
import numpy as np


class PINNScaler:
    """   Skálázza a változó paramétereket, hogy könnyebben átsúlyozhassuk a loss-t, ne legyen ellipszoid a felület:
        Args: t, S, A, v, (float) ezeknek minimuma és maximuma, kivéve ugye a t-t aminek 0.
        
     Attributes:
        bounds (dict): A változók nevét (str) és a hozzájuk tartozó (min, max) tuple határértékeket.
     """
    def __init__(self, t_max, S_min=0.0, S_max=5.0, v_min=0.0001, v_max=1.0, A_min=0.1, A_max=5.0):
        self.bounds = {
            'tau': (0.0, t_max),
            'S_K': (S_min, S_max),
            'v':   (v_min, v_max),
            'A_K': (A_min, A_max)
        }

    def scale(self, x_physical, var_name):
        """ a skálázó függvény, 
    Args:  x_physical (tensor), var_name(str)
    dict-ből kiszedi a tuple-t
    returns: x_physical-t berakja [-1, 1] köze:
    minmax: 2*[0, 1]-1 = [-1, 1]
        
"""
        x_min, x_max = self.bounds[var_name]
        return 2.0 * (x_physical - x_min) / (x_max - x_min) - 1.0


class LAAF_GELU(nn.Module):
    """ próbálkoztam geluval is, az egyik cikk alapján, de nem működödtt"""
    def __init__(self):
        super(LAAF_GELU, self).__init__()
       
        self.n = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x):
        
        return F.gelu(self.n * x)

class LAAF_Tanh(nn.Module):
    """nn.Module kiegészítve egy tanulható n paraméterrel
    
    """
    def __init__(self):
        super(LAAF_Tanh, self).__init__()
        self.n = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x):
        """ LAAF_ tanh kiszámolása
        Args: x (tensor)
        Returns: tanh(nx) (tensor)
        """
        return torch.tanh(self.n * x)


class HestonAsianPINN(nn.Module):
    """ skálázza, felépíti a neurális hálót kibővítve az nn.Modulet 
        
        Args: scaler (class)
        Attributes: PINNScaler (class) szerint scalel, ezt meghívja, és a Sequential egymásba ágyazza a függvényeket
    """
    def __init__(self, scaler=None):
        super(HestonAsianPINN, self).__init__()
        self.scaler = scaler
        self.net = nn.Sequential(
            nn.Linear(4, 256), LAAF_Tanh(),
            nn.Linear(256, 256), LAAF_Tanh(),
            nn.Linear(256, 256), LAAF_Tanh(),
            nn.Linear(256, 256), LAAF_Tanh(),
            nn.Linear(256, 256), LAAF_Tanh(),
            nn.Linear(256, 128), LAAF_Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, tau, S, v, A):
        """ scaleli a tensorokat, majd a cat segítsésgével összefűzi az Nx1 tensorokat Nx4-é ez lesz x
            args: S, v , tau, A (tensor)
            return: becsült opciós ár: Nx1 tensor
            Note: A forward egy kötött név, az nn.Moduleban U = model.__call__(tau, S, v, A), ahol a __call__-ban return self.forward van
        """
        
        if self.scaler is not None:
            t_s = self.scaler.scale(tau, 'tau')
            S_s = self.scaler.scale(S, 'S_K')
            v_s = self.scaler.scale(v, 'v')
            A_s = self.scaler.scale(A, 'A_K')
        else:
            t_s, S_s, v_s, A_s = tau, S, v, A
        
        x = torch.cat([t_s, S_s, v_s, A_s], dim=1)
        return self.net(x)


class OptionSampler:
    """ mintavételez a vizsgálandő térből, 
        Args:  t, S, A, v, (float) ezeknek minimuma és maximuma, kivéve ugye a t-t aminek 0, valamint, az eszköz amin fut: device (class a torch.device)
        Attributes: Definiálja a teret, amelyből mintát kell vételezni, valamint a helyet ahol futtatja
        
    """
    def __init__(self, device, tau_max, S_min=0.0, S_max=5.0, v_min=0.0001, v_max=1.0, A_min=0.1, A_max=5.0):
        self.device = device
        self.tau_max = tau_max
        self.S_min, self.S_max = S_min, S_max
        self.v_min, self.v_max = v_min, v_max
        self.A_min, self.A_max = A_min, A_max

    def sample_interior(self, n_points):
        """
        Belső pontok generálása Latin Hypercube Sampling segítségével, a 4 dimenziós állapottéren. 
        . A függvén egy nemlineáris sűrítési transzformációt alkalmaz S és A esetében a K=1.55 kötési árfolyam (ez most a strike) körül. 
        Erre azért van szükség, mert az opciós árazásnál a payoff
        töréspontja környékén a PDE megoldása numerikusan nehezebb (nagyon magas Gamma), 
        így a PINN hálónak sűrűbb tanítási rácsra van szüksége ebben a régióban. Ezt úgy teszi meg, hogy
        egy harmadfokú polinomiális sűrítést rak be
        Args: n_points (int): batch mérete.

        Returns: tuple: tau, S, v, A PyTorch tenzorok, mindegyik [n_points, 1] alakkal,
                   a device-ra betöltve.
        """
        sampler = qmc.LatinHypercube(d=4)
        u = sampler.random(n=n_points)
        
        tau = torch.tensor(u[:, 0] * self.tau_max, device=self.device).view(n_points, 1).float()
        v = torch.tensor(self.v_min + u[:, 1] * (self.v_max - self.v_min), device=self.device).view(n_points, 1).float()
        
        S_raw = self.S_min + u[:, 2] * (self.S_max - self.S_min)
        A_raw = self.A_min + u[:, 3] * (self.A_max - self.A_min)
        #legyen 0.35
        c = 0.4 

        # A a harmadfokú polinom transzformáció, ami tiszteletben tartja a peremeket
        S_mag = S_raw + c * (1.55 - S_raw) * (S_raw - self.S_min) * (self.S_max - S_raw)
        A_mag = A_raw + c * (1.55 - A_raw) * (A_raw - self.A_min) * (self.A_max - A_raw)
        # Clamp a biztonság kedvéért a sűrítés miatt, ha véletlen kimenne 0, és 3 ból
        S = torch.clamp(torch.tensor(S_mag, device=self.device).view(n_points, 1).float(), self.S_min, self.S_max)
        A = torch.clamp(torch.tensor(A_mag, device=self.device).view(n_points, 1).float(), self.A_min, self.A_max)
        
        return tau, S, v, A

    def sample_initial_condition(self, n_points):
        """ Lejárati peremfeltételen pontok mintavételezése
            3 dimes, hiszen tau már 0
            args: n_points (int)
            Returns: tau, S, v, A (tensor) (tuple-ben)
        """
        sampler = qmc.LatinHypercube(d=3)
        u = sampler.random(n=n_points)
        tau = torch.zeros((n_points, 1), dtype=torch.float32, device=self.device)
        S = torch.tensor(self.S_min + u[:, 0] * (self.S_max - self.S_min), device=self.device).view(-1, 1).float()
        v = torch.tensor(self.v_min + u[:, 1] * (self.v_max - self.v_min), device=self.device).view(-1, 1).float()
        A = torch.tensor(self.A_min + u[:, 2] * (self.A_max - self.A_min), device=self.device).view(-1, 1).float()
        return tau, S, v, A

    def sample_boundary_S(self, n_points, S_value):
        """felső, és alsó peremfeltételeken
            3 dimes, hiszen S= 3 , vagy S = 0
            args: n_points (int)
            Returns: tau, S, v, A (tensor) (tuple-ben)
        """
        sampler = qmc.LatinHypercube(d=3)
        u = sampler.random(n=n_points)
        
        tau = torch.tensor(u[:, 0] * self.tau_max, device=self.device).view(-1, 1).float()
        v = torch.tensor(self.v_min + u[:, 1] * (self.v_max - self.v_min), device=self.device).view(-1, 1).float()
        A = torch.tensor(self.A_min + u[:, 2] * (self.A_max - self.A_min), device=self.device).view(-1, 1).float()
        
        S = torch.full_like(tau, S_value, device=self.device)
        return tau, S, v, A


def calculate_heston_asian_pde(model, tau, S, v, A, params):
    """ a params dictionary-ből kiolvassa az értékeket, majd a model (class HestonAsianPINN) segítségével
        és kiszámolja a PDE hibát (latex)
        Args: tau, S, v, A, (tensor) params (dict), model (class HestonAsianPINN def forward)
        Returns: pde hiba, 
    """
    r, kappa, theta, sigma, rho = params['r'], params['kappa'], params['theta'], params['sigma'], params['rho']
    U = model(tau, S, v, A)
    ones = torch.ones_like(U) #láncszabály szerinti szorzáshoz kell, hogy 1-es súllyal szerepeljen a láncban a derivált
    
    dU_dS = torch.autograd.grad(U, S, grad_outputs=ones, create_graph=True)[0]
    dU_dv = torch.autograd.grad(U, v, grad_outputs=ones, create_graph=True)[0]
    dU_dtau = torch.autograd.grad(U, tau, grad_outputs=ones, create_graph=True)[0]
    dU_dA = torch.autograd.grad(U, A, grad_outputs=ones, create_graph=True)[0]

    d2U_dS2 = torch.autograd.grad(dU_dS, S, grad_outputs=ones, create_graph=True)[0]
    d2U_dv2 = torch.autograd.grad(dU_dv, v, grad_outputs=ones, create_graph=True)[0]
    d2U_dSdv = torch.autograd.grad(dU_dS, v, grad_outputs=ones, create_graph=True)[0]

    res = (-dU_dtau + 0.5 * v * (S**2) * d2U_dS2 + rho * sigma * v * S * d2U_dSdv + 
           0.5 * (sigma**2) * v * d2U_dv2 + r * S * dU_dS + kappa * (theta - v) * dU_dv + S * dU_dA - r * U)
    return res

def calculate_s_zero_loss(model, tau_bc, v_bc, A_bc, params, T_maturity):
    """ latex dokumentum szerint kiszámolja a hibát az S = 0 pontokban, softplus van a peremfeltétel helyett, mert softplus deriválható és hasonlít a maxhoz.
    Args: model (callable: HestonAsianPinn ) tau_bc, v_bc, A_bc (tensor), params (dict), T_maturity (float) 
    Returns: MSE (float)
    """
    
    r = params['r']
    K = params['K'] 
    S0 = torch.zeros_like(tau_bc)
    U_pred = model(tau_bc, S0, v_bc, A_bc)
    payoff = F.softplus((A_bc / T_maturity) - K, beta=20.0) 
    U_true = torch.exp(-r * tau_bc) * payoff
    return torch.mean((U_pred - U_true)**2)

def calculate_payoff_loss(model, tau_ic, S_ic, v_ic, A_ic, T_maturity, params):
    """ latex dokumentum szerint kiszámolja a hibát a t = T pontban softplus van a peremfeltétel helyett, mert softplus deriválható és hasonlít a maxhoz.
    Args: model (callable: HestonAsianPinn ) tau_bc, v_bc, A_bc (tensor), params (dict), T_maturity (float) 
    Returns: MSE (float)
    """
    K = params['K']
    U_pred = model(tau_ic, S_ic, v_ic, A_ic)
    payoff = F.softplus((A_ic / T_maturity) - K, beta=20.0)
    return torch.mean((U_pred - payoff) ** 2)
    
def calculate_neumann_bc_loss(model, tau_bc, S_bc, v_bc, A_bc):
    """ latex dokumentum szerint kiszámolja a hibát az S = 3 pontban. Ez a neumann peremfeltétel, Gamma = 0
    Args: model (callable: HestonAsianPinn ) tau_bc, v_bc, A_bc (tensor), params (dict), T_maturity (float) 
    Returns: MSE (float)
    """
    U = model(tau_bc, S_bc, v_bc, A_bc)
    ones = torch.ones_like(U)
    dU_dS = torch.autograd.grad(U, S_bc, grad_outputs=ones, create_graph=True)[0]
    d2U_dS2 = torch.autograd.grad(dU_dS, S_bc, grad_outputs=ones, create_graph=True)[0]
    return torch.mean(d2U_dS2 ** 2)
