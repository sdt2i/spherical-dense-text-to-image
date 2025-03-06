import os
import math

import imageio
import torch
import numpy as np
from torchvision.utils import save_image
import torch.nn.functional as F

class Tesellation(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device

    def get_tangent_images(self, erp_image, fov_x, fov_y, list_of_angles):
        batch_size, num_channels, height, width = erp_image.shape
        theta_phis = torch.FloatTensor(list_of_angles).to(self.device)
        num_angles = theta_phis.shape[0]
        # get rays from the perspective
        perspective_rays = self.get_rays_at_z(fov_x=fov_x, fov_y=fov_y, erp_h=height, erp_w=width, batch_size=batch_size)
        # rotate the rays to the sphere
        r_mat_persp_2_sphere = self.get_inverse_rotation_for_center(theta=theta_phis[:, 0], phi=theta_phis[:,1])
        out_h, out_w = perspective_rays.shape[1], perspective_rays.shape[2]
        perspective_rays = perspective_rays.unsqueeze(1).unsqueeze(-1)  # [B, 20, H, W, 3, 1]
        perspective_rays = perspective_rays.expand(batch_size, num_angles, out_h, out_w, 3, 1)
        
        r_mat_persp_2_sphere = r_mat_persp_2_sphere.view(1, num_angles, 1, 1, 3, 3) # [1, 20, 3, 3]
        rays_on_sphere = torch.matmul(r_mat_persp_2_sphere, perspective_rays) # [B, 20, H, W, 3, 1]
        rays_on_sphere = rays_on_sphere.view(batch_size*num_angles, out_h, out_w, 3)
        # # project rays from cartesian to spherical cooridnates
        rays_spherical_coords = self.cartesian_2_spherical(rays_on_sphere)
        # # project rays from spherical cooridnates to pixel coordinates
        rays_equi_coords = self.spherical_2_equi(rays_spherical_coords, height, width)

        # # normalize the pixel coordinates
        rays_equi_coords[..., 0] = rays_equi_coords[..., 0] - (width-1)/2.0
        rays_equi_coords[..., 0] = rays_equi_coords[..., 0] / ((width-1)/2.0)
        rays_equi_coords[..., 1] = rays_equi_coords[..., 1] - (height-1)/2.0
        rays_equi_coords[..., 1] = rays_equi_coords[..., 1] / ((height-1)/2.0)
        # repeat the erp image by the number of angles
        erp_image = erp_image.unsqueeze(1).expand(batch_size, num_angles, num_channels, height, width)
        erp_image = erp_image.view(batch_size*num_angles, num_channels, height, width)
        # # sample the ERP image
        perspective = F.grid_sample(erp_image,
                            grid=rays_equi_coords.clamp(min=-1.0, max=1.0),
                            mode='bilinear',
                            align_corners=True)
        return perspective.view(batch_size, num_angles, num_channels, out_h, out_w)

    def remap_tangents(self, tangents, list_of_angles, erp_h, erp_w):
        batch_size, num_angles, num_channels, p_height, p_width = tangents.shape
        theta_phis = torch.FloatTensor(list_of_angles).to(self.device)
        assert num_angles == theta_phis.shape[0], "Number of angles should be same as tangents"
        points_on_sphere = self._get_unit_rays_on_sphere(batch_size, erp_h, erp_w)
        points_on_sphere = points_on_sphere.unsqueeze(1).unsqueeze(-1)
        points_on_sphere = points_on_sphere.expand(batch_size, num_angles, erp_h, erp_w, 3, 1)
        # Rotate points from sphere to perspective cameras
        r_mat_persp_2_sphere = self.get_inverse_rotation_for_center(theta=theta_phis[:, 0], phi=theta_phis[:, 1])
        r_mat_sphere_2_persp = r_mat_persp_2_sphere.inverse().view(1, num_angles, 1, 1, 3, 3)
        unit_rays_in_perspective = torch.matmul(r_mat_sphere_2_persp, points_on_sphere)
        # # Compute KMatrix of the perspective camera
        fx = erp_w / (2.0*math.pi)
        cx = (p_width-1.0) / 2.0
        cy = (p_height-1.0) / 2.0
        k_matrix = torch.FloatTensor([[fx, 0, cx], [0, fx, cy], [0, 0, 1]]).to(self.device)
        k_matrix = k_matrix.view(1, 1, 1, 1, 3, 3)
        k_matrix = k_matrix.expand(batch_size, num_angles, erp_h, erp_w, 3, 3)
        # Project points on the persepective cameras
        xy_hom = torch.matmul(k_matrix, unit_rays_in_perspective).squeeze(-1)
        # # Computer
        negative_points = unit_rays_in_perspective[..., 2, 0].le(0)
        # xy_hom[]
        x_pts = (xy_hom[..., 0] / xy_hom[..., 2])
        y_pts = (xy_hom[..., 1] / xy_hom[..., 2])
        x_pts = (x_pts - ((p_width-1)/2)) / ((p_width-1)/2)
        y_pts = (y_pts - ((p_height-1)/2)) / ((p_height-1)/2)
        x_pts[negative_points] = -2
        y_pts[negative_points] = -2
        grid = torch.cat([x_pts.view(batch_size, num_angles, erp_h, erp_w, 1),
                         y_pts.view(batch_size,  num_angles, erp_h, erp_w, 1)], dim=-1)
        grid = grid.view(batch_size*num_angles, erp_h, erp_w, 2)
        mask_x = torch.logical_and(x_pts.ge(-1), x_pts.le(1))
        mask_y = torch.logical_and(y_pts.ge(-1), y_pts.le(1))
        mask = torch.logical_and(mask_x, mask_y).view(batch_size, num_angles, 1, erp_h, erp_w)
        # import pdb;pdb.set_trace()
        warped_img = F.grid_sample(tangents.view(batch_size*num_angles, num_channels, p_height, p_width),
                                   grid=grid.clamp(min=-1, max=1), mode="bilinear", align_corners=True)
        warped_img = warped_img.view(batch_size,  num_angles, num_channels, erp_h, erp_w)
        mask_float = mask.float()
        warped_img = (warped_img * mask_float).sum(dim=1)
        mask_weight = mask_float.sum(dim=1)
        warped_img = warped_img / mask_weight.clamp(min=1)
        # # current_mask = mask.repeat(1, c, 1, 1)
        # # current_erp[erp_mask.gt(0).repeat(1, c, 1, 1)
        # #             ] = erp_bank[erp_mask.gt(0).repeat(1, c, 1, 1)]
        # # current_erp[current_mask] = warped_img[current_mask]
        return warped_img

    def _get_unit_rays_on_sphere(self, batch_size, erp_h, erp_w):
        x_y_locs = self.get_xy_coords(batch_size=batch_size, height=erp_h, width=erp_w)
        sph_locs = self.equi_2_spherical(x_y_locs)
        xyz_locs = self.spherical_2_cartesian(sph_locs).expand(batch_size, erp_h, erp_w, 3)  # B, H, W, 3
        return xyz_locs

    def get_xy_coords(self, batch_size, height, width):
        device = self.device
        x_locs = torch.linspace(0, width-1, width).view(1, width, 1)
        y_locs = torch.linspace(0, height-1, height).view(height, 1, 1)
        x_locs, y_locs = map(lambda x: x.to(device), [x_locs, y_locs])
        x_locs, y_locs = map(lambda x: x.expand(
            height, width, 1), [x_locs, y_locs])
        xy_locs = torch.cat([x_locs, y_locs], dim=2)
        xy_locs = xy_locs.unsqueeze(0).expand(batch_size, height, width, 2)
        return xy_locs

    def cartesian_2_spherical(self, input_points, normalized=False):
        last_coord_one = False
        if input_points.shape[-1] == 1:
            input_points = input_points.squeeze(-1)
            last_coord_one = True
        if not normalized:
            input_points = self.normalize_3d_vectors(input_points)
        x_c, y_c, z_c = torch.split(
            input_points, split_size_or_sections=1, dim=-1)
        r = torch.sqrt(x_c**2 + y_c**2 + z_c**2)
        theta = torch.atan2(y_c, x_c)
        phi = torch.acos(z_c/r)
        mask1 = theta.lt(0)
        theta[mask1] = theta[mask1] + 2*math.pi
        # mask2 = theta.lt(0)
        # theta[mask2] = theta[mask2] + 2*math.pi
        spherical_coords = torch.cat(
            [theta, phi, torch.ones_like(theta)], dim=-1)
        # spherical to equi
        return spherical_coords

    def spherical_2_equi(self, spherical_coords, height, width):
        last_coord_one = False
        if spherical_coords.shape[-1] == 1:
            spherical_coords = spherical_coords.squeeze(-1)
            last_coord_one = True
        spherical_coords = torch.split(
            spherical_coords, split_size_or_sections=1, dim=-1)
        theta, phi = spherical_coords[0], spherical_coords[1]
        x_locs = (width-1) * (1 - theta/(2.0*math.pi))
        y_locs = phi*(height-1)/math.pi
        xy_locs = torch.cat([x_locs, y_locs], dim=-1)
        if last_coord_one:
            xy_locs = xy_locs.unsqueeze(-1)

        return xy_locs

    def equi_2_spherical(self, equi_coords, radius=1):
        """
        """
        batch_size, height, width, _ = equi_coords.shape
        device = equi_coords.device
        input_shape = equi_coords.shape
        assert input_shape[-1] == 2, 'last coordinate should be 2'
        x_locs, y_locs = torch.split(
            tensor=equi_coords, dim=-1, split_size_or_sections=1)
        x_locs = x_locs.clamp(min=0, max=width-1)
        y_locs = y_locs.clamp(min=0, max=height-1)
        theta = (-2*math.pi / (width-1)) * x_locs + 2*math.pi
        phi = (math.pi/(height-1))*(y_locs)
        spherical_coords = torch.cat(
            [theta, phi, torch.ones_like(theta).mul(radius)], dim=-1)

        return spherical_coords.to(device)

    def spherical_2_cartesian(self, spherical_coords):  # checked
        input_shape = spherical_coords.shape
        assert input_shape[-1] in [2,
                                   3], 'last dimension of input should be 3 or 2'
        coordinate_split = torch.split(
            spherical_coords, split_size_or_sections=1, dim=-1)
        theta, phi = coordinate_split[:2]
        if input_shape[-1] == 3:
            rad = coordinate_split[2]
        else:
            rad = torch.ones_like(theta).to(theta.device)
        x_locs = rad * torch.sin(phi) * torch.cos(theta)
        y_locs = rad * torch.sin(phi) * torch.sin(theta)
        z_locs = rad * torch.cos(phi)
        xyz_locs = torch.cat([x_locs, y_locs, z_locs], dim=-1)
        return xyz_locs

    def get_inverse_rotation_for_center(self, phi, theta):
        device = self.device
        """"cpp
        EMatrix3x3 getInvRotationForCenter(ScalarDouble centerPhi, ScalarDouble centerTheta) const
        {
            EVector3 rotationVector{0., 0., Pi_2 - centerTheta};
            ScalarDouble vectorLength{rotationVector.norm()};

            EMatrix3x3 Rz;
            if (vectorLength > EPS)
            {
                Eigen::AngleAxisd rzAngle{vectorLength, rotationVector / vectorLength};
                Rz = rzAngle.toRotationMatrix();
            }
            else
                Rz.setIdentity();

            rotationVector << centerPhi, 0., 0.;
            vectorLength = rotationVector.norm();

            EMatrix3x3 Rx;
            if (vectorLength > EPS)
            {
                Eigen::AngleAxisd rxAngle{vectorLength, rotationVector / vectorLength};
                Rx = rxAngle.toRotationMatrix();
            }
            else
                Rx.setIdentity();

            return Rz.transpose() * Rx.transpose();
        }
        """
        rz = self.get_rotation_z(90 - theta)
        rx = self.get_rotation_x(phi)
        # return (rz.permute(1, 0) @ rx.permute(1, 0)).to(device)
        return torch.bmm(rz.permute(0, 2, 1), rx.permute(0, 2, 1)).to(device)

    def get_rays_at_z(self, fov_x, fov_y, erp_h, erp_w, batch_size=1):
        device = self.device
        b = batch_size
        focal_len = erp_w / (2*math.pi)
        p_h = int(2 * focal_len * math.tan(math.radians(fov_y/2)))
        p_w = int(2 * focal_len * math.tan(math.radians(fov_x/2)))

        y_locs = torch.linspace(0, p_h-1, p_h).to(device)
        x_locs = torch.linspace(0, p_w-1, p_w).to(device)
        x_locs = x_locs.view(1, 1, p_w, 1).expand(b, p_h, p_w, 1)
        y_locs = y_locs.view(1, p_h, 1, 1).expand(b, p_h, p_w, 1)
        # compute rays
        x_locs = (x_locs - (p_w-1)*0.5) / focal_len
        y_locs = (y_locs - (p_h-1)*0.5) / focal_len
        ones = torch.ones(b, p_h, p_w, 1).to(device)
        rays = torch.cat([x_locs, y_locs, ones], dim=3)
        # import pdb; pdb.set_trace()
        rays = self.normalize_3d_vectors(rays)
        return rays

    def normalize_3d_vectors(self, input_points, p=2, eps=1e-12):
        p_norm = torch.norm(input_points, p=p, dim=-1,
                            keepdim=True).clamp(min=eps)
        return input_points / p_norm

    def _read_image(self, input_erp):
        img = torch.from_numpy(imageio.imread(input_erp)).float()
        img = img.permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        # img = F.interpolate(img.to(self.device), size=(self.erp_h, self.erp_w), antialias=False, mode='bilinear', align_corners=False)
        return img

    def _save_image(self, img, img_path):
        save_image(tensor=img.cpu(), fp=img_path)

    # def get_rotation_x(self, angle, device):
    #     device = self.device
    #     angle = math.radians(angle)
    #     sin, cos = math.sin(angle), math.cos(angle)
    #     r_mat = torch.eye(3).to(device)
    #     r_mat[1, 1] = cos
    #     r_mat[1, 2] = -sin
    #     r_mat[2, 1] = sin
    #     r_mat[2, 2] = cos
    #     return r_mat

    # def get_rotation_z(self, angle, device):
    #     device = self.device
    #     # print('***', angle)
    #     angle = math.radians(angle)
    #     sin, cos = math.sin(angle), math.cos(angle)
    #     r_mat = torch.eye(3).to(device)
    #     r_mat[0, 0] = cos
    #     r_mat[0, 1] = -sin
    #     r_mat[1, 0] = sin
    #     r_mat[1, 1] = cos
    #     return r_mat
    def get_rotation_x(self, angle):
        device = self.device
        b_size = angle.shape[0]
        angle = torch.deg2rad(angle)
        sin, cos = torch.sin(angle), torch.cos(angle)
        r_mat = torch.eye(3).view(1, 3, 3).to(device)
        r_mat = r_mat.repeat(b_size, 1, 1)
        r_mat[:, 1, 1] = cos.view(-1)
        r_mat[:, 1, 2] = -sin.view(-1)
        r_mat[:, 2, 1] = sin.view(-1)
        r_mat[:, 2, 2] = cos.view(-1)
        return r_mat

    def get_rotation_z(self, angle):
        device = self.device
        b_size = angle.shape[0]
        angle = torch.deg2rad(angle)
        sin, cos = torch.sin(angle), torch.cos(angle)
        r_mat = torch.eye(3).view(1, 3, 3).to(device)
        r_mat = r_mat.repeat(b_size, 1, 1)
        r_mat[:, 0, 0] = cos.view(-1)
        r_mat[:, 0, 1] = -sin.view(-1)
        r_mat[:, 1, 0] = sin.view(-1)
        r_mat[:, 1, 1] = cos.view(-1)
        return r_mat

def get_theta_phi_values(level=0):
    """
        TODO: create a mapping from level to theta, phi values
    """
    theta_phi_vals = [
        [36., -67.5],
        [108., -67.5],
        [180., -67.5],
        [252., -67.5],
        [324., -67.5],
        [72., -22.5],
        [144., -22.5],
        [216., -22.5],
        [288., -22.5],
        [360., -22.5],
        [36., 22.5],
        [108., 22.5],
        [180., 22.5],
        [252., 22.5],
        [324., 22.5],
        [36., 67.5],
        [108., 67.5],
        [180., 67.5],
        [252., 67.5],
        [324., 67.5]]
    # our convention for phi is [0, pi] and theta is [2pi, 0]
    theta_phi_values = [[t_p[0], t_p[1]+90] for t_p in theta_phi_vals]
    return theta_phi_values
    
def get_horizontal_theta_phi_vals():
    theta_phi_vals = []
    for a in range(180, -180, -5):
        theta_phi_vals.append([a, 0])
    theta_phi_values = [[t_p[0], t_p[1] + 90] for t_p in theta_phi_vals]
    return theta_phi_values

def main():
    WIDTH, HEIGHT = 1024, 512
    file_name = "desert3.png"
    sphere_utils = Tesellation()
    erp_image = sphere_utils._read_image(f"./{file_name}")
    erp_image = F.interpolate(erp_image, size=(
        HEIGHT, WIDTH), antialias=False, mode='bilinear', align_corners=False)
    batch_size, channels, erp_h, erp_w = erp_image.shape
    # Compute tangents
    tangents = sphere_utils.get_tangent_images(erp_image=erp_image, fov_x=120, fov_y=120, list_of_angles=get_horizontal_theta_phi_vals())
    print(tangents.shape)
    # Remap tangents
    #erp_image_est = sphere_utils.remap_tangents(
    #    tangents=tangents, list_of_angles=get_theta_phi_values(), erp_h=erp_h, erp_w=erp_w)
    # Save results
    #tangents_path = f"./results_{file_name}/tangent_images"
    tangents_path = "./frames"
    erps_path = f"./results_{file_name}/erps_images"
    os.makedirs(tangents_path, exist_ok=True)
    #os.makedirs(erps_path, exist_ok=True)
    #save_image(erp_image.cpu(), os.path.join(erps_path, "erp_org.png"))
    #save_image(erp_image_est.cpu(), os.path.join(erps_path, "erp_est.png"))
    for itr, b in enumerate(range(tangents.shape[1])):
        c_tangent = tangents[0, b, ...].cpu()
        save_image(c_tangent.cpu(),
                   os.path.join(tangents_path, f"{str(itr).zfill(4)}.png"))

if __name__ == '__main__':
    main()
