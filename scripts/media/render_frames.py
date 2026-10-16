"""Step B (SAPIEN 3, e.g. `pip install sapien==3.0.1`): render the Allegro at the retargeted joints next to the tracked hand."""
import argparse, numpy as np, sapien, imageio.v2 as iio
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
p = argparse.ArgumentParser(); p.add_argument("--npz"); p.add_argument("--out"); p.add_argument("--title", default=""); p.add_argument("--robot_title", default="Allegro hand (retargeted)")
p.add_argument("--cam", type=float, nargs=3, default=[0.62, 0.0, 0.50]); p.add_argument("--mirror", action="store_true")
a = p.parse_args()
z = np.load(a.npz, allow_pickle=True)
scene = sapien.Scene()
scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([0.3, 0.6, -1], [1.0, 1.0, 1.0]); scene.add_directional_light([-0.5, -0.3, -0.6], [0.5, 0.5, 0.55])
art = scene.create_urdf_loader().load(str(z["urdf"]))
art.set_root_pose(sapien.Pose([0, 0, 0.35], [0.695, 0, -0.718, 0]))
order = [ [j.get_name() for j in art.get_active_joints()].index(n) for n in z["names"] ]
cam = scene.add_camera("cam", 720, 720, np.deg2rad(30), 0.01, 10)
tgt = np.array([0.0, 0.0, 0.415]); eye = np.array(a.cam)
f = tgt - eye; f /= np.linalg.norm(f); l = np.cross([0, 0, 1], f); l /= np.linalg.norm(l); u = np.cross(f, l)
M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = f, l, u, eye
cam.entity.set_pose(sapien.Pose(M))
BONES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
COL = ["#e15759"]*4 + ["#4e79a7"]*4 + ["#59a14f"]*4 + ["#f28e2b"]*4 + ["#b07aa1"]*4
fig = plt.figure(figsize=(7.2, 7.2), dpi=100); ax = fig.add_axes([0, 0, 1, 0.92], projection="3d")
fig.text(0.5, 0.95, a.title or "operator hand (Quest 3 hand tracking)", ha="center", fontsize=17)
kp = z["kp"]; lo, hi = kp.reshape(-1, 3).min(0), kp.reshape(-1, 3).max(0); c = (lo + hi) / 2; r = (hi - lo).max() / 2 * 0.62
frames = []
for k, q in zip(kp, z["q_sim"]):
    ax.cla()
    for (i, j), col in zip(BONES, COL): ax.plot(*zip(k[i], k[j]), color=col, lw=4)
    ax.scatter(*k.T, s=14, c="k")
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
    ax.view_init(elev=12, azim=0); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_axis_off(); fig.canvas.draw()
    left = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    qq = np.zeros(len(order)); qq[order] = q; art.set_qpos(qq)
    scene.update_render(); cam.take_picture()
    rgb = (np.clip(cam.get_picture("Color")[..., :3], 0, 1) * 255).astype(np.uint8)
    seg = cam.get_picture("Segmentation")[..., 1]
    rgb[seg == 0] = 255
    import cv2; cv2.putText(rgb, a.robot_title, (int(360 - 7.6 * len(a.robot_title)), 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    frames.append(np.concatenate([left, rgb], 1))
if a.out.endswith(".png"): iio.imwrite(a.out, frames[len(frames)//2])
else:
    import os; os.makedirs(a.out, exist_ok=True)
    for i, fr in enumerate(frames): iio.imwrite(f"{a.out}/{i:04d}.png", fr)
print("wrote", a.out, len(frames))
