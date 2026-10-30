"""Physical attachment checks for the rigid PhotoHipExo training variant."""
import unittest

import mujoco as mj
import numpy as np

from myoassist_utils.env_spec import EnvSpec


class PhotoHipExoBindingTest(unittest.TestCase):
    def test_binding_through_hip_motion(self):
        for msk in ("myolegs22", "myolegs26"):
            with self.subTest(msk=msk):
                model = mj.MjModel.from_xml_string(EnvSpec(msk, "PhotoHipExo_L1").validate().compose())
                data = mj.MjData(model)
                prefix = "PhotoHipExo_L1_"
                waist = model.body(prefix + "waist").id
                pelvis = model.body("pelvis").id
                self.assertEqual(model.body_parentid[waist], pelvis)
                self.assertAlmostEqual(sum(model.body_mass[i] for i in range(model.nbody)
                                           if model.body(i).name.startswith(prefix)), 3.15, places=5)
                self.assertFalse(any("passive_abduction" in model.joint(i).name for i in range(model.njnt)))
                self.assertEqual([model.actuator(i).name for i in range(model.nu - 2, model.nu)],
                                 ["Exo_R", "Exo_L"])
                # Cuffs must have a constant transform to their femur at every pose.
                cuff_poses = {}
                for angle in (0, .6, -.3):
                    data.qpos[:] = model.qpos0
                    for side in ("r", "l"):
                        j = model.joint("hip_flexion_" + side)
                        data.qpos[j.qposadr[0]] = angle if side == "r" else -angle
                    mj.mj_forward(model, data)
                    wrot = data.xmat[waist].reshape(3, 3)
                    if angle == 0:
                        np.testing.assert_allclose(wrot[:, 1], [0, 0, 1], atol=1e-12)
                    for side, sign in (("r", 1), ("l", -1)):
                        hip = model.joint("hip_flexion_" + side).id
                        # Housing centers lie laterally along the human hip shaft.
                        motor = data.xpos[waist] + wrot @ np.array([0, 0, sign * .186828658284019])
                        delta = motor - data.xanchor[hip]
                        np.testing.assert_allclose(np.cross(delta, data.xaxis[hip]), 0, atol=1e-12)
                        femur = model.body("femur_" + side).id
                        cuff = model.geom(prefix + "distal_plate_" + side).id
                        frot = data.xmat[femur].reshape(3, 3)
                        local = np.r_[frot.T @ (data.geom_xpos[cuff] - data.xpos[femur]),
                                      (frot.T @ data.geom_xmat[cuff].reshape(3, 3)).ravel()]
                        if angle == 0:
                            cuff_poses[side] = local
                        np.testing.assert_allclose(local, cuff_poses[side], atol=1e-12)
                data.ctrl[-2:] = [1, -1]
                mj.mj_forward(model, data)
                np.testing.assert_allclose(data.actuator_force[-2:], [20, -20])


class HousingPayloadTest(unittest.TestCase):
    def test_added_mass_stays_in_housing(self):
        from MyHipExo.mass_properties import calculate

        vertices = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
        # Closed cube with outward-facing triangles.
        faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                          [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                          [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
        parts = {'battery': (vertices + [1, 0, 0], faces),
                 'motor': (vertices, faces),
                 'cuff_l': (vertices, faces), 'cuff_r': (vertices, faces)}
        groups = {'waist': ['battery', 'motor'], 'link_l': ['cuff_l'], 'link_r': ['cuff_r']}
        params = dict(total_mass=2.55, link_cuff_mass=.15, hip_half_width=.2)
        before, _ = calculate(parts, groups, params, lambda side: np.zeros(3))
        params.update(total_mass=3.15, housing_added_mass=.6)
        after, report = calculate(parts, groups, params, lambda side: np.zeros(3))
        self.assertAlmostEqual(after['waist']['mass_kg'] - before['waist']['mass_kg'], .6)
        for side in ('l', 'r'):
            self.assertEqual(after['link_' + side], before['link_' + side])
        center = (2.25 * np.array([.5, 0, 0]) + .6 * np.array([1, 0, 0])) / 2.85
        np.testing.assert_allclose(after['waist']['center_body_frame_m'], center)
        # Unit cubes have central inertia m/6 along every axis.
        expected = np.eye(3) * (2.85 / 6)
        for mass, x in ((1.125, 0.), (1.725, 1.)):
            d = np.array([x, 0, 0]) - center
            expected += mass * (d @ d * np.eye(3) - np.outer(d, d))
        np.testing.assert_allclose(after['waist']['inertia_at_center_body_axes_kg_m2'], expected, atol=1e-15)
        self.assertAlmostEqual(report['assembled_neutral_pose']['mass_kg'], 3.15)


if __name__ == "__main__":
    unittest.main()
