import unittest

from src.vision.pose_depth_metrics import PoseDepthMetricEngine


_DEFAULT_SHOULDER = object()


class PoseDepthMetricEngineHandRaiseTests(unittest.TestCase):
    def setUp(self):
        self.engine = PoseDepthMetricEngine()
        self.shoulder_center = (100.0, 100.0)
        self.shoulder = (100.0, 100.0)
        self.torso_length = 100.0

    def is_raised(self, wrist, elbow, shoulder=_DEFAULT_SHOULDER):
        return self.engine._is_hand_raised(
            wrist_point=wrist,
            elbow_point=elbow,
            shoulder_point=self.shoulder if shoulder is _DEFAULT_SHOULDER else shoulder,
            shoulder_center=self.shoulder_center,
            torso_length=self.torso_length,
        )

    def test_clear_vertical_arm_raise_counts(self):
        self.assertTrue(self.is_raised(wrist=(108.0, 36.0), elbow=(106.0, 70.0)))

    def test_bent_classroom_hand_raise_counts(self):
        self.assertTrue(self.is_raised(wrist=(112.0, 64.0), elbow=(118.0, 94.0)))

    def test_low_head_near_shoulder_pose_does_not_count(self):
        self.assertFalse(self.is_raised(wrist=(92.0, 84.0), elbow=(88.0, 128.0)))

    def test_reading_or_holding_book_far_from_shoulder_does_not_count(self):
        self.assertFalse(self.is_raised(wrist=(300.0, 40.0), elbow=(190.0, 160.0)))

    def test_missing_elbow_or_shoulder_does_not_count(self):
        self.assertFalse(self.is_raised(wrist=(108.0, 36.0), elbow=None))
        self.assertFalse(self.is_raised(wrist=(108.0, 36.0), elbow=(106.0, 70.0), shoulder=None))


if __name__ == '__main__':
    unittest.main()
