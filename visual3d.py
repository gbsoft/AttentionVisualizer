import os, sys
import time
import numpy as np
import pyvista as pv
from pyvista import examples
import pyvistaqt as pvqt

# static 3d
def demo_basic_3d():
    # mesh=examples.load_ant()
    mesh=examples.load_hexbeam()
    p=pv.Plotter()
    p.add_mesh(mesh)
    p.show()

def load_point_cloud(points):
    N, dim = points.shape
    assert N>0 and dim in [3, 4]
    if dim==3:
        point_cloud = points 
        pdata = pv.PolyData(point_cloud)
        pdata['orig_sphere'] = np.zeros(len(points))
    else:
        point_values = points[:,3]
        point_cloud = points[:,:3]
        pdata = pv.PolyData(point_cloud)
        pdata['orig_sphere'] = point_values

    return pdata

def demo_basic_point_cloud_3d():
    show_basic_point_cloud(np.random.random((100, 3)))

def show_basic_point_cloud(points):
    pdata = load_point_cloud(points)
    sphere = pv.Sphere(radius=0.2, phi_resolution=10, theta_resolution=10)
    pc = pdata.glyph(scale=False, geom=sphere, orient=False)
    pc.plot(cmap='Reds')

# dynamic 3d
cnt=0
begin=time.time()
very_begin=begin
def show_dynamic_point_cloud(points): 
    pdata = load_point_cloud(points) 
    sphere = pv.Sphere(radius=0.2, phi_resolution=10, theta_resolution=10)
    mesh = pdata.glyph(scale=False, geom=sphere, orient=False)

    p = pvqt.BackgroundPlotter()
    p.add_mesh(mesh)
    p.show_bounds(grid=not True, location='back')
    p.view_isometric()

    def update():
        global cnt, begin, very_begin
        cnt+=1
        p.camera.Azimuth(0.1)
        p.update()
        if cnt%100==0:
            end=time.time()
            print(f'Frame={cnt}, Time={end-very_begin:.2f}s, FPS={100/(end-begin):.2f}')
            begin=time.time()

    p.add_callback(update, 1)
    pv.Plotter().show(window_size=[1,1])

# Overall
def show_point_cloud(points, dynamic=0):
    if dynamic:
        show_dynamic_point_cloud(points)
    else:
        show_basic_point_cloud(points)
    
if __name__=='__main__':
    points = np.random.random((100, 4)) * 10
    show_point_cloud(points, 0)
