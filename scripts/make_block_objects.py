"""Write box meshes that look like the real DOFBOT blocks in the SRE's 400x400 view into assets/objects/blocks/.

Size is matched in image space, not metres: in the rectified real view a block covers a median 155 x 74 px (p25-p75:
142-169 x 65-90, measured from 153 real Mask R-CNN masks). Calibrated by rendering: boxes of 0.10-0.13 x 0.045-0.07 m
covered a median 121 x 62 px in the sim top view, so 0.125-0.16 x 0.055-0.083 m gives the SRE the real picture. The
long axis lies along x, so boxes rest flat, like the real blocks, and survive Environment.remove_flat_objs.

    python scripts/make_block_objects.py      # 16 boxes, seeded
"""
import os

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "objects", "blocks")
FACES = [(1, 2, 3), (1, 3, 4), (5, 8, 7), (5, 7, 6), (1, 5, 6), (1, 6, 2),
         (2, 6, 7), (2, 7, 3), (3, 7, 8), (3, 8, 4), (4, 8, 5), (4, 5, 1)]


def box_obj(dx, dy, dz):
    x, y, z = dx / 2, dy / 2, dz / 2
    v = [(-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z), (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)]
    return "".join(f"v {a:.5f} {b:.5f} {c:.5f}\n" for a, b, c in v) + "".join(f"f {i} {j} {k}\n" for i, j, k in FACES)


def main(n=16, seed=0):
    rng = np.random.RandomState(seed)
    os.makedirs(OUT, exist_ok=True)
    for i in range(n):
        dx, dy, dz = rng.uniform(0.125, 0.16), rng.uniform(0.055, 0.083), rng.uniform(0.05, 0.075)
        with open(os.path.join(OUT, f"block_{i:02d}.obj"), "w") as f:
            f.write(f"# box {dx:.3f} x {dy:.3f} x {dz:.3f} m\n" + box_obj(dx, dy, dz))
    print(f"wrote {n} boxes to {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
