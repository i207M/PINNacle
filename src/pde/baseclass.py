import deepxde as dde
import numpy as np

DEFAULT_NUM_DOMAIN_POINTS = 8192
DEFAULT_NUM_BOUNDARY_POINTS = 2048
DEFAULT_NUM_TEST_POINTS = 8192
DEFAULT_NUM_INITIAL_POINTS = 2048


class BasePDE():

    def __init__(self):
        self.pde = None
        self.bcs = None
        self.geom = None
        self.bbox = None
        self.loss_config = []
        self.output_config = None

        self.num_domain_points = DEFAULT_NUM_DOMAIN_POINTS
        self.num_boundary_points = DEFAULT_NUM_BOUNDARY_POINTS
        self.num_test_points = DEFAULT_NUM_TEST_POINTS

        self.ref_sol = None
        self.ref_data = None

        self.recommend_net = None

    @property
    def input_dim(self):
        return self.geom.dim

    @property
    def output_dim(self):
        if self.output_config is None:
            raise ValueError("output_config not set")
        return len(self.output_config)

    @output_dim.setter
    def output_dim(self, value):
        if self.output_config is None:
            self.output_config = [{'name': f'y_{i+1}'} for i in range(value)]
        else:
            assert self.output_dim == value, "output_config and output_dim not matched"

    @property
    def num_pde(self):
        return sum(map((lambda c: c['type'] == 'pde'), self.loss_config))

    @property
    def num_gepinn(self):
        return sum(map((lambda c: c['type'] == 'gepinn'), self.loss_config))

    @property
    def num_boundary(self):
        return sum(map((lambda c: c['type'] == 'boundary'), self.loss_config))

    @property
    def num_loss(self):
        return len(self.loss_config)

    def trans_time_data_to_dataset(self, datapath):
        data = self.ref_data
        slice = (data.shape[1] - self.input_dim + 1) // self.output_dim
        assert slice * self.output_dim == data.shape[1] - self.input_dim + 1, "Data shape is not multiple of pde.output_dim"
        
        with open(datapath, "r") as f:
            def extract_time(string):
                index = string.find("t=")
                if index == -1:
                    return None
                return float(string[index+2:].split(' ')[0])
            t = None
            for line in f.readlines():
                if line.startswith('%') and line.count('@') == slice * self.output_dim:
                    t = line.split('@')[1:]
                    t = list(map(extract_time, t))
            if t is None or None in t: 
                raise ValueError("Reference Data not in Comsol format or does not contain time info")
            t = np.array(t[::self.output_dim])

        t, x0 = np.meshgrid(t, data[:, 0])
        list_x = [x0.reshape(-1)]
        for i in range(1, self.input_dim - 1):
            list_x.append(np.stack([data[:, i] for _ in range(slice)]).T.reshape(-1))
        list_x.append(t.reshape(-1))
        for i in range(self.output_dim):
            list_x.append(data[:, self.input_dim - 1 + i::self.output_dim].reshape(-1))
        self.ref_data = np.stack(list_x).T

    def load_ref_data(self, datapath, transform_fn=None, t_transpose=False):
        self.ref_data = np.loadtxt(datapath, comments="%").astype(np.float32)
        if t_transpose:  # originally used only in BaseTimePDE, but needed from some TimePDE using BasePDE as baseclass.
            self.trans_time_data_to_dataset(datapath)
        if transform_fn is not None:
            self.ref_data = transform_fn(self.ref_data)

    def set_pdeloss(self, names=None, num=1):
        if names is not None:
            self.loss_config += [{"name": name, "type": 'pde'} for name in names]
        else:
            self.loss_config += [{"name": f"pde_{i}", "type": 'pde'} for i in range(num)]

    def add_bcs(self, config, geom=None):
        geom = geom if geom is not None else self.geom

        if self.bcs is None:
            self.bcs = []
        for bc in config:
            if bc.get('name') is None:
                bc['name'] = bc['type'] + ('' if bc['type'] == 'ic' else 'bc') + f"_{len(self.bcs) + 1}"
            if bc['type'] == 'dirichlet':
                self.bcs.append(dde.DirichletBC(geom, bc['function'], bc['bc'], component=bc['component']))
            elif bc['type'] == 'robin':
                self.bcs.append(dde.RobinBC(geom, bc['function'], bc['bc'], component=bc['component']))
            elif bc['type'] == 'ic':
                self.bcs.append(dde.IC(geom, bc['function'], bc['bc'], component=bc['component']))
            elif bc['type'] == 'operator':
                self.bcs.append(dde.OperatorBC(geom, bc['function'], bc['bc']))
            elif bc['type'] == 'neumann':
                self.bcs.append(dde.NeumannBC(geom, bc['function'], bc['bc'], component=bc['component']))
            elif bc['type'] == 'periodic':
                self.bcs.append(dde.PeriodicBC(geom, bc['component_x'], bc['bc'], component=bc['component']))
            elif bc['type'] == 'pointset':
                self.bcs.append(dde.PointSetBC(bc['points'], bc['values'], component=bc['component']))
            else:
                raise ValueError(f"Unknown bc type: {bc['type']}")
            self.loss_config.append({'name': bc['name'], 'type': 'boundary'})

    def training_points(self, domain=DEFAULT_NUM_DOMAIN_POINTS, boundary=DEFAULT_NUM_BOUNDARY_POINTS, test=DEFAULT_NUM_TEST_POINTS, mul=1):
        self.num_domain_points = domain * mul
        self.num_boundary_points = boundary * mul
        self.num_test_points = test * mul

    def check(self):
        if self.pde is None:
            raise ValueError("PDE could not be None")
        if self.geom is None:
            raise ValueError("geometry could not be None")
        if self.output_config is None:
            raise ValueError("output config could not be None, please set output dim or output config")
        if self.bbox is None:
            raise ValueError("bbox could not be None")
        if self.num_pde == 0:
            raise ValueError("No pde loss specified")

        for i in range(self.num_pde):
            if self.loss_config[i]['type'] != 'pde':
                raise ValueError("All PDE loss should be set before Boundary loss to avoid potential issues with methods like NTK")

    def use_gepinn(self):
        pde_original = self.pde
        def pde_wrapper(x, u): # add regularize terms to pde function
            res = pde_original(x, u)

            if not isinstance(res, (list, tuple)):  # convert single pde loss to list
                res = [res]
            elif isinstance(res, tuple):  # convert tuple_like pde loss to list
                res = list(res)

            for r in res:  # unsqueeze pde loss of shape (batch_size,) to shape (batch_size, 1)
                if r.dim() == 1:
                    r = r.unsqueeze(dim=-1)
                else:
                    assert r.dim() == 2 and r.shape[1] == 1, "improper pde residue shape"

            res_g = []
            for j in range(self.input_dim):
                for i in range(len(res)):
                    res_g.append(dde.grad.jacobian(res[i], x, i=0, j=j))

            return res + res_g

        self.pde = pde_wrapper

        config = []
        for j in range(self.input_dim):
            for i in range(self.num_pde):
                config.append({"name": self.loss_config[i]['name'] + f"_grad{j}", "type": "gepinn"})
        self.loss_config = self.loss_config[:self.num_pde] + config + self.loss_config[self.num_pde:]

    def create_model(self, net):
        self.check()
        self.net = net
        self.data = dde.data.PDE(
            self.geom,
            self.pde,
            self.bcs,
            num_domain=self.num_domain_points,
            num_boundary=self.num_boundary_points,
            num_test=self.num_test_points,
        )
        self.model = dde.Model(self.data, net)
        self.model.pde = self
        return self.model


class BaseTimePDE(BasePDE):

    def __init__(self):
        super().__init__()
        self.geomtime = None
        self.num_initial_points = DEFAULT_NUM_INITIAL_POINTS

    @property
    def input_dim(self):
        return self.geomtime.dim

    def add_bcs(self, config):
        super().add_bcs(config, self.geomtime)

    def load_ref_data(self, datapath, transform_fn=None, t_transpose=True):
        super(BaseTimePDE, self).load_ref_data(datapath, transform_fn, t_transpose)

    def training_points(
        self,
        domain=DEFAULT_NUM_DOMAIN_POINTS,
        boundary=DEFAULT_NUM_BOUNDARY_POINTS,
        initial=DEFAULT_NUM_INITIAL_POINTS,
        test=DEFAULT_NUM_TEST_POINTS,
        mul=1,
    ):
        self.num_domain_points = domain * mul
        self.num_boundary_points = boundary * mul
        self.num_initial_points = initial * mul
        self.num_test_points = test * mul

    def create_model(self, net):
        self.check()
        self.net = net
        self.data = dde.data.TimePDE(
            self.geomtime,
            self.pde,
            self.bcs,
            num_domain=self.num_domain_points,
            num_boundary=self.num_boundary_points,
            num_initial=self.num_initial_points,
            num_test=self.num_test_points
        )
        self.model = dde.Model(self.data, net)
        self.model.pde = self
        return self.model
