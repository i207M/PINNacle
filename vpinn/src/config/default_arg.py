import json
import os

class SolverArg:
    def __init__(self, **kwargs):
        # Defining all possible parameters and setting them to None
        self.layer_size = None
        self.neuron_type = None
        self.grid_num = None #
        self.coef = None #
        self.Q = None #
        self.test_fcn_num = None #
        self.epoch_num = None
        self.model_name = None
        self.data = None
        self.layers = None
        self.u_ref = None
        self.grid_data = None

        # Updating the parameters based on the provided kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def set_layer_size(self, new_layer_size):
        self.layer_size = new_layer_size
        
    def set_coef(self, loss_weight):
        for i in range(min(len(loss_weight), len(self.coef))):
            self.coef[i] = loss_weight[i]
    
    def set_epoch_num(self, epoch_num):
        self.epoch_num = epoch_num
    
    def to_dict(self):
        return {key: value for key, value in vars(self).items() if value is not None}

class ArgContainer:
    def __init__(self):
        self.Burgers1d = SolverArg(layer_size=[2, 15, 15, 15, 1], neuron_type='tanh', grid_num=[6, 6], coef=[1e-3, 1e-3], Q=10, test_fcn_num=6, epoch_num=10000, model_name='Burgers1d', data='burgers1d.dat')
        self.Burgers2d = SolverArg(layer_size=[3, 30, 30, 30, 2], neuron_type='tanh', grid_num=[6, 6, 6], coef=[1, 1, 10], Q=10, test_fcn_num=5, epoch_num=10000, model_name='Burgers2d', data='burgers2d.dat', layers=[0])
        self.Poisson1 = SolverArg(layer_size=[2, 15, 15, 15, 1], neuron_type='tanh', grid_num=[4, 4], coef=[10], Q=10, test_fcn_num=5, epoch_num=10000, model_name='Poisson1', u_ref='self.u')
        self.Poisson2d = SolverArg(layer_size=[2, 15, 15, 15, 1], neuron_type='tanh', grid_num=[32, 32], coef=[0.4] * 5, Q=10, test_fcn_num=5, epoch_num=10000, model_name='Poisson2d', data='poisson_classic.dat')
        self.Poisson_boltzmann2d = SolverArg(layer_size=[2, 20, 20, 20, 1], neuron_type='tanh', grid_num=[32, 32], coef=[0.00001] * 5, Q=10, test_fcn_num=5, epoch_num=10000, model_name='Poisson-Boltzmann2d', data='poisson_boltzmann2d.dat')
        self.Poisson3d = SolverArg(layer_size=[3, 20, 20, 20, 1], neuron_type='tanh', grid_num=[8, 8, 8], coef=[1] * 2, Q=10, test_fcn_num=5, epoch_num=10000, model_name='Poisson3d', data='poisson_3d.dat', grid_data='self.griddata', layers='self.layers')
        self.Poisson2d_Many_subdomains = SolverArg(layer_size=[2, 20, 20, 20, 1], neuron_type='tanh', grid_num=[8, 8], coef=[1, 0.1], Q=10, test_fcn_num=5, epoch_num=10000, model_name='Poisson2d_Many_subdomains', data='poisson_manyarea.dat')
        self.Heat2d_Varying_Source = SolverArg(layer_size=[3, 20, 20, 20, 1], neuron_type='tanh', grid_num=[4, 4, 4], coef=[1, 1], Q=10, test_fcn_num=5, epoch_num=10000, model_name='Heat2d_Varying_Source', data='heat_darcy.dat', layers=[0, 0.5, 2, 3.5])
        self.Heat_Multi_scale = SolverArg(layer_size=[3, 20, 20, 20, 1], neuron_type='tanh', grid_num=[4] * 3, coef=[0.1] * 2, Q=10, test_fcn_num=5, epoch_num=10000, model_name='Heat_Multi_scale', data='heat_multiscale.dat', layers=[0, 1, 2, 3])
        self.HeatComplex = SolverArg(layer_size=[3, 20, 20, 20, 1], neuron_type='tanh', grid_num=[4] * 3, coef=[1] * 19, Q=10, test_fcn_num=5, epoch_num=10000, model_name='HeatComplex', data='heat_complex.dat', layers=[0, 1, 2, 3])
        self.HeatLongTime = SolverArg(layer_size=[3, 20, 20, 20, 1], neuron_type='tanh', grid_num=[4] * 3, coef=[1] * 3, Q=10, test_fcn_num=5, epoch_num=10000, model_name='HeatLongTime', data='heat_longtime.dat', layers=[0, 1, 2, 3])
        self.NSEquation_LidDriven = SolverArg(layer_size=[2, 20, 20, 20, 3], neuron_type='tanh', grid_num=[4] * 2, coef=[0.1] * 5 , Q=10, test_fcn_num=5, epoch_num=10000, model_name='2D_Lid_Driven')
        self.NS_Back_Step = SolverArg(layer_size=[2, 20, 20, 20, 3], neuron_type='tanh', grid_num=[4] * 2, coef=[0.1] * 13, Q=10, test_fcn_num=5, epoch_num=10000, model_name='NS_Back_Step', data='ns_4_obstacle.dat')
        self.NSEquation_Long = SolverArg(layer_size=[3, 20, 20, 20, 3], neuron_type='tanh', grid_num=[4] * 3, coef=[0.1] * 6, Q=10, test_fcn_num=5, epoch_num=10000, model_name='NSEquation_Long', data='ns_long.dat', layers=[0, 1, 2, 3])
        self.WaveEquation1D = SolverArg(layer_size=[2, 20, 20, 20, 1], neuron_type='tanh', grid_num=[8] * 2, coef=[0.1] * 6, Q=10, test_fcn_num=5, epoch_num=10000, model_name='WaveEquation1D')
        self.WaveHeterogeneous = SolverArg(layer_size=[3, 20, 20, 20, 1], neuron_type='tanh', grid_num=[8] * 3, coef=[0.1] * 3, Q=10, test_fcn_num=5, epoch_num=10000, model_name='WaveHeterogeneous', data='wave_darcy.dat', layers=[0, 1, 2, 3])
        self.WaveEquation2D_Long = SolverArg(layer_size=[3, 30, 30, 30, 1], neuron_type='tanh', grid_num=[8] * 3, coef=[0.1, 0.1, 0.1], Q=10, test_fcn_num=5, epoch_num=10000, model_name='WaveEquation2D_Long', layers=[0, 1, 2, 3])
        self.GrayScottEquation = SolverArg(layer_size=[3, 20, 20, 20, 2], neuron_type='tanh', grid_num=[4] * 3, coef=[0.1] * 2, Q=10, test_fcn_num=5, epoch_num=10000, model_name='GrayScottEquation', layers=[0, 10, 20, 30], data='grayscott.dat')
        self.KuramotoSivashinskyEquation = SolverArg(layer_size=[2, 20, 20, 20, 1], neuron_type='tanh', grid_num=[8] * 2, coef=[1] * 1, Q=10, test_fcn_num=5, epoch_num=10000, model_name='KuramotoSivashinskyEquation', layers=[0, 10, 20, 30], data='Kuramoto_Sivashinsky.dat')
        self.PoissonInv = SolverArg(layer_size=[2, 20, 20, 20, 1], neuron_type='tanh', grid_num=[4] * 2, coef=[10] * 2, Q=10, test_fcn_num=5, epoch_num=10000, model_name='PoissonInv')
        self.HeatInv = SolverArg(layer_size=[3, 20, 20, 20, 1], neuron_type='tanh', grid_num=[8] * 3, coef=[0.1] * 4, Q=10, test_fcn_num=5, epoch_num=10000, model_name='HeatInv', layers=[0, 0.5, 1])
        
    def to_json(self, file_path='src/config/default_arg.json'):
        arg_map = {
            'Burgers1d': self.Burgers1d.to_dict(),
            'Burgers2d': self.Burgers2d.to_dict(),
            'Poisson1': self.Poisson1.to_dict(),
            'Poisson2d': self.Poisson2d.to_dict(),
            'Poisson_boltzmann2d': self.Poisson_boltzmann2d.to_dict(),
            'Poisson3d': self.Poisson3d.to_dict(),
            'Poisson2d_Many_subdomains': self.Poisson2d_Many_subdomains.to_dict(),
            'Heat2d_Varying_Source': self.Heat2d_Varying_Source.to_dict(),
            'Heat_Multi_scale': self.Heat_Multi_scale.to_dict(),
            'HeatComplex': self.HeatComplex.to_dict(),
            'HeatLongTime': self.HeatLongTime.to_dict(),
            'NSEquation_LidDriven': self.NSEquation_LidDriven.to_dict(),
            'NS_Back_Step': self.NS_Back_Step.to_dict(),
            'NSEquation_Long': self.NSEquation_Long.to_dict(),
            'WaveEquation1D': self.WaveEquation1D.to_dict(),
            'WaveHeterogeneous': self.WaveHeterogeneous.to_dict(),
            'WaveEquation2D_Long': self.WaveEquation2D_Long.to_dict(),
            'GrayScottEquation': self.GrayScottEquation.to_dict(),
            'KuramotoSivashinskyEquation': self.KuramotoSivashinskyEquation.to_dict(),
            'PoissonInv': self.PoissonInv.to_dict(),
            'HeatInv': self.HeatInv.to_dict(),
        }
        with open(file_path, 'w') as f:
            json.dump(arg_map, f, indent=4)
        return arg_map

argContainer = ArgContainer()

if __name__ == "__main__":
    current_script_path = os.path.abspath(__file__)
    current_script_dir = os.path.dirname(current_script_path)
    os.chdir(current_script_dir)
    argContainer.to_json('default_arg.json')