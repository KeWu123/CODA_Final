import ast
import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_PATH = ROOT / 'code' / 'analyze_sliceeq_posterior_gate.py'
UTILITY_PATH = ROOT / 'code' / 'utils' / 'sliceeq_posterior.py'
PROTOCOL_PATH = (
    ROOT / 'research' / 'experiments' / 'h7_slice_profile_reacquisition' /
    'h7_4_posterior_commutation_gate_protocol.md')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def parser_literal_defaults(path):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and \
                isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    defaults = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute) or \
                node.func.attr != 'add_argument' or not node.args or \
                not isinstance(node.args[0], ast.Constant):
            continue
        for keyword in node.keywords:
            if keyword.arg != 'default':
                continue
            try:
                defaults[node.args[0].value] = ast.literal_eval(keyword.value)
            except (ValueError, TypeError):
                if isinstance(keyword.value, ast.Name) and \
                        keyword.value.id in constants:
                    defaults[node.args[0].value] = constants[
                        keyword.value.id]
    return defaults


class SliceEqPosteriorGateContractTest(unittest.TestCase):
    def test_previous_gate_and_training_sources_are_unchanged(self):
        expected = {
            'code/analyze_sliceeq_gates.py':
                'cf031624ca86d1d2078ac50b71a8e3fdfd77cccbc1ace4399e81bee03e6e3d35',
            'code/utils/sliceeq_gates.py':
                '7646963d8628d77de8ef17e95c1d8863ab8083896be191b3583bcbbbc4b78283',
            'code/train_sliceeq_occ.py':
                'f9391d1979eba4b87ec5fc6368bbc376201bcf66e83feb1106ef36efb0ec93e5',
            'code/utils/sliceeq_occ.py':
                'de2fc77c40de4d543e6aa682589aaae652887de9429742d08feacc97e50d6080',
            'code/utils/sliceeq.py':
                '44a956a92eccdbb2109034a05ac5ec72f190b33f740ad95c1ed8b505bae168f7',
            'code/dataloaders/sliceeq_dataset.py':
                '9cc39cd6ed373e22ec854340c7975868025c2bf74223370c1eeb69e928fc19d5',
        }
        for relative_path, expected_hash in expected.items():
            self.assertEqual(
                sha256(ROOT / relative_path), expected_hash, relative_path)

    def test_analysis_is_read_only_and_reuses_h7_3_reference(self):
        source = ANALYSIS_PATH.read_text(encoding='utf-8')
        tree = ast.parse(source, filename=str(ANALYSIS_PATH))
        called_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    called_names.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    called_names.add(node.func.attr)
        self.assertNotIn('backward', called_names)
        self.assertNotIn('step', called_names)
        self.assertNotIn('save', called_names)
        self.assertNotIn('train', called_names)
        self.assertIn("'analysis_kind': 'read_only_no_training'", source)
        self.assertIn('model, checkpoint_format = h73._load_model(', source)
        self.assertIn('_reference_reproduction(', source)
        self.assertIn('REFERENCE_REPRODUCTION_ATOL = 1e-6', source)

    def test_locked_variants_thresholds_and_defaults_are_explicit(self):
        source = ANALYSIS_PATH.read_text(encoding='utf-8')
        defaults = parser_literal_defaults(ANALYSIS_PATH)
        self.assertIn("'hard_lcc'", source)
        self.assertIn("'raw_posterior'", source)
        self.assertIn("'topology_gated_posterior'", source)
        self.assertIn('MAX_EXACT_SUPPORT_BRIER_RATIO = 0.85', source)
        self.assertIn('MIN_RESIDUAL_PEARSON = 0.50', source)
        self.assertIn('MAX_OUTSIDE_RESIDUAL_MASS = 0.15', source)
        self.assertIn('MAX_FULL_IMAGE_BRIER_RATIO = 1.05', source)
        self.assertEqual(defaults['--seed'], 1337)
        self.assertEqual(defaults['--labeled_slices'], 191)
        self.assertEqual(defaults['--batch_size'], 12)
        self.assertEqual(defaults['--max_labeled_batches'], 16)
        self.assertEqual(defaults['--sliceeq_sigma_min'], 0.45)
        self.assertEqual(defaults['--sliceeq_sigma_max'], 0.85)
        self.assertEqual(defaults['--sliceeq_phase_min'], -0.25)
        self.assertEqual(defaults['--sliceeq_phase_max'], 0.25)
        self.assertEqual(
            defaults['--h7_3_reference_json'],
            '../model/SliceEqOcc_PROMISE12_7_labeled/analysis/'
            'h7_3_gates_iter23000.json')

    def test_protocol_and_implementation_define_the_same_brier(self):
        protocol = PROTOCOL_PATH.read_text(encoding='utf-8')
        utility = UTILITY_PATH.read_text(encoding='utf-8')
        self.assertIn(
            'mean_c (O_candidate(c,v) - O_gt(c,v))^2', protocol)
        self.assertIn('.square().mean(dim=1)', utility)


if __name__ == '__main__':
    unittest.main()
