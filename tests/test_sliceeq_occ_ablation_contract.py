import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / 'code' / 'train_sliceeq_occ_ablation.py'
RUNNER = ROOT / 'run_sliceeq_occ_ablation.sh'
PIPELINE = ROOT / 'run_sliceeq_occ_ablation_pipeline.sh'
M0_M1_PIPELINE = ROOT / 'run_sliceeq_occ_m0_m1_pipeline.sh'
M2_M3_PIPELINE = ROOT / 'run_sliceeq_occ_m2_m3_pipeline.sh'
PAPER_PIPELINE = ROOT / 'run_sliceeq_occ_paper_ablation_pipeline.sh'
FACTORIAL_PIPELINE = ROOT / 'run_sliceeq_occ_factorial_pipeline.sh'
DOC = ROOT / 'docs' / 'SLICEEQ_OCC_ABLATIONS.md'


class SliceEqOccAblationContractTest(unittest.TestCase):
    def test_ablation_choices_are_locked(self):
        tree = ast.parse(TRAIN.read_text(encoding='utf-8'))
        source = TRAIN.read_text(encoding='utf-8')
        self.assertIn("choices=['baseline', 'baseline_36', 'image_only',", source)
        self.assertIn("components['labeled_target_mode'] == 'hard'", source)
        self.assertIn("components['unlabeled_target_mode'] == 'center'", source)
        self.assertIn("not components['use_labeled_reacq']", source)
        self.assertIn("if components['teacher_uses_neighbors']", source)
        self.assertIn('ema_model(unlabeled_images)', source)
        self.assertIn("if components['reacquire_unlabeled_image']", source)
        self.assertIn('unlabeled_reacquired_images = unlabeled_images', source)
        self.assertIsNotNone(tree)

    def test_incremental_chain_adds_exactly_one_component(self):
        namespace = {}
        tree = ast.parse(TRAIN.read_text(encoding='utf-8'))
        assignment = next(
            node for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name)
                    and target.id == 'ABLATION_COMPONENTS'
                    for target in node.targets))
        exec(compile(ast.Module(body=[assignment], type_ignores=[]),
                     str(TRAIN), 'exec'), namespace)
        components = namespace['ABLATION_COMPONENTS']

        self.assertFalse(components['baseline']['use_labeled_reacq'])
        self.assertFalse(
            components['baseline']['reacquire_unlabeled_image'])
        self.assertTrue(
            components['image_only']['reacquire_unlabeled_image'])
        self.assertEqual(
            components['aligned_occ']['unlabeled_target_mode'],
            'fractional')
        self.assertEqual(
            components['full']['labeled_target_mode'], 'fractional')
        self.assertEqual(
            components['full']['unlabeled_target_mode'], 'fractional')
        self.assertEqual(
            components['no_labeled_reacq'], components['aligned_occ'])

    def test_paper_chain_is_compute_and_view_matched(self):
        namespace = {}
        tree = ast.parse(TRAIN.read_text(encoding='utf-8'))
        assignment = next(
            node for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name)
                    and target.id == 'ABLATION_COMPONENTS'
                    for target in node.targets))
        exec(compile(ast.Module(body=[assignment], type_ignores=[]),
                     str(TRAIN), 'exec'), namespace)
        components = namespace['ABLATION_COMPONENTS']
        chain = ('baseline_36', 'image_only_36', 'hard_targets', 'full')
        for name in chain:
            self.assertTrue(components[name]['use_labeled_reacq'])
        self.assertFalse(
            components['baseline_36']['reacquire_labeled_image'])
        self.assertFalse(
            components['baseline_36']['reacquire_unlabeled_image'])
        self.assertTrue(
            components['image_only_36']['reacquire_labeled_image'])
        self.assertEqual(
            components['hard_targets']['labeled_target_mode'], 'hard')
        self.assertEqual(
            components['hard_targets']['unlabeled_target_mode'], 'hard')

    def test_factorial_controls_change_only_fractional_target_location(self):
        namespace = {}
        tree = ast.parse(TRAIN.read_text(encoding='utf-8'))
        assignment = next(
            node for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name)
                    and target.id == 'ABLATION_COMPONENTS'
                    for target in node.targets))
        exec(compile(ast.Module(body=[assignment], type_ignores=[]),
                     str(TRAIN), 'exec'), namespace)
        components = namespace['ABLATION_COMPONENTS']
        left = components['occ_l_only']
        right = components['occ_u_only']
        for key in (
                'reacquire_unlabeled_image', 'teacher_uses_neighbors',
                'use_labeled_reacq', 'reacquire_labeled_image'):
            self.assertEqual(left[key], right[key])
            self.assertTrue(left[key])
        self.assertEqual(left['labeled_target_mode'], 'fractional')
        self.assertEqual(left['unlabeled_target_mode'], 'hard')
        self.assertEqual(right['labeled_target_mode'], 'hard')
        self.assertEqual(right['unlabeled_target_mode'], 'fractional')

    def test_runner_uses_shared_label7_protocol(self):
        source = RUNNER.read_text(encoding='utf-8')
        self.assertIn('OCC_ABLATION="${OCC_ABLATION:-image_only}"', source)
        self.assertIn('SliceEqOccIncremental_${OCC_ABLATION}', source)
        self.assertIn(
            'baseline|baseline_36|image_only|image_only_36|aligned_occ|hard_targets|occ_l_only|occ_u_only|full|no_labeled_reacq',
            source)
        self.assertIn('--labelnum 7', source)
        self.assertIn('--max_iterations 30000', source)
        self.assertIn('*7_labeled*|*label7*|*7label*', source)
        self.assertIn('cd "${ROOT}/code"', source)

    def test_document_names_required_runs(self):
        source = DOC.read_text(encoding='utf-8')
        for name in ('baseline', 'image_only', 'aligned_occ', 'full',
                     'hard_targets', 'PosteriorOcc'):
            self.assertIn(name, source)

    def test_shared_pipeline_trains_then_tests_in_order(self):
        source = PIPELINE.read_text(encoding='utf-8')
        self.assertIn('for stage in "${STAGE_LIST[@]}"', source)
        self.assertIn('bash "${ROOT}/run_sliceeq_occ_ablation.sh"', source)
        self.assertIn('python -u test_sliceeq_occ.py', source)
        self.assertIn('unet_best_model.pth', source)
        self.assertIn('performance.txt', source)
        self.assertIn('ALLOW_EXISTING="${ALLOW_EXISTING:-0}"', source)

    def test_stage_wrappers_lock_the_requested_pairs(self):
        m0_m1 = M0_M1_PIPELINE.read_text(encoding='utf-8')
        m2_m3 = M2_M3_PIPELINE.read_text(encoding='utf-8')
        self.assertIn('STAGES="baseline image_only"', m0_m1)
        self.assertIn('PIPELINE_NAME="m0_m1"', m0_m1)
        self.assertIn('STAGES="aligned_occ full"', m2_m3)
        self.assertIn('PIPELINE_NAME="m2_m3"', m2_m3)

    def test_paper_and_factorial_wrappers_lock_stage_order(self):
        paper = PAPER_PIPELINE.read_text(encoding='utf-8')
        factorial = FACTORIAL_PIPELINE.read_text(encoding='utf-8')
        self.assertIn(
            'STAGES="baseline_36 image_only_36 hard_targets full"', paper)
        self.assertIn('PIPELINE_NAME="paper_main"', paper)
        self.assertIn('STAGES="occ_l_only occ_u_only"', factorial)
        self.assertIn('PIPELINE_NAME="factorial_lu"', factorial)


if __name__ == '__main__':
    unittest.main()
