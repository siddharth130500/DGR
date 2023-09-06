import numpy as np
from tqdm import tqdm

# Velodyne HDL-64E data at https://pdf.directindustry.com/pdf/velodynelidar/hdl-64e-datasheet/182407-676099.html

class DGR:
    def __init__(self, pc_path):
        self.delta_phis = 0.08
        self.omega_bot = -24.8
        self.phi_left = 0
        self.FOV = 26.8
        self.Nv = 64
        self.delta_omegas = self.FOV/self.Nv
        self.HOV = 360
        self.Nh = int(self.HOV/self.delta_phis)
        self.pc = self.read_lidar(pc_path) # Original point cloud
        self.Pi = [np.empty((0, 4))] * (self.Nv + 1) # Original point cloud in clusters
        self.Pi_star = [] # Resampled point cloud
        self.angle_set = set()

        # Resample parameters
        # self.Ndv = 60
        # self.delta_phid = 0.1
        self.Ndv = 40
        self.delta_phid = 0.5
        self.delta_omegad = self.FOV / self.Ndv
        self.Ndh = int(self.HOV/self.delta_phid)

    def read_lidar(self, pc_path):
        return np.fromfile(pc_path, dtype=np.float32).reshape(-1,4)

    def create_clusters(self):
        """
        Split 3D points from Pi into Nv elevation angle
        clusters Pi = {C1, C2, ...CNv}
        """
        omega = []
        for i in range(self.pc.shape[0]):
            omega.append(np.arcsin(self.pc[i][2] / np.sqrt(self.pc[i][0] ** 2 + self.pc[i][1] ** 2 + \
                                                           self.pc[i][2] ** 2)))
        omega = (np.array(omega)) * 180 / np.pi

        Gv = np.linspace(-24.8, 2, num=self.Nv+1)
        print("Creating Clusters")
        for i in tqdm(range(omega.shape[0])):
            idx = np.argmin(np.abs(Gv - omega[i]))
            self.Pi[idx] = np.concatenate((self.Pi[idx],self.pc[i].reshape(1,4)),axis=0)

    def resample(self):
        w = 3
        Tnorm = 0.25
        Gdv = np.linspace(-24.8,2,num=self.Ndv+1) * np.pi/180
        Gdh = np.linspace(-180, 180, num=self.Ndh+1) * np.pi/180
        for i in tqdm(range(self.Nv+1)):
            ptsn = self.Pi[i]
            ptsw = np.empty((0,4), dtype=np.float32)
            for k in range(-int(w/2),int(w/2)+1):
                if i+k>=len(self.Pi) or i+k<0:
                    continue
                ptsw = np.concatenate((ptsw,self.Pi[i+k]),axis=0)

            for ii in range(ptsn.shape[0]):
                ptsb = ptsw[np.where(np.absolute(np.linalg.norm(ptsn[ii][:3]) - np.linalg.norm(ptsw[:,:3], axis=1)) < Tnorm)][:,:3]
                norm_mu = np.mean(np.linalg.norm(ptsb, axis=1))
                pout = ptsn[ii][:3]
                omega_out = np.arcsin(pout[2] / np.sqrt(pout[0] ** 2 + pout[1] ** 2 + \
                                                           pout[2] ** 2))
                idx = np.argmin(np.abs(Gdv - omega_out))
                omega_out = Gdv[idx]
                phi_out = np.sign(pout[1]) * np.arccos(pout[0] / np.sqrt(pout[0] ** 2 + pout[1] ** 2))
                idx = np.argmin(np.abs(Gdh - phi_out))
                phi_out = Gdh[idx]
                if (omega_out,phi_out) not in self.angle_set and norm_mu>0:
                    r = norm_mu
                    self.angle_set.add((omega_out,phi_out))
                else:
                    r = np.linalg.norm(pout)
                pout = np.array([r * np.cos(omega_out) * np.cos(phi_out), r * np.cos(omega_out) * np.sin(phi_out),
                                 r * np.sin(omega_out), ptsn[ii][3]])
                self.Pi_star += [pout.reshape(1, 4)]

        np.array(self.Pi_star).reshape(-1,4).astype('float32').tofile('/home/sid/pts.bin') # The path to the resampled point cloud


if __name__ == "__main__":

    path = '/home/sid/kitti_detection/training/velodyne/000019.bin' # The path to the original point cloud

    dgr = DGR(path)
    dgr.create_clusters()
    dgr.resample()






