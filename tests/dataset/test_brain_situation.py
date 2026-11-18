"""Situation diagnostics and checkpoint selection on synthetic ticks, with no optimisation run."""
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from nnslr_tools import brain_replay, brain_targets, brain_train


def drive(partitions, kph):
    """A tiny drive with numeric inputs, speed labels and individually addressable place cells."""
    n = len(partitions)
    inputs = {name: np.zeros((n, size), dtype=np.float32) for name, size in brain_targets.INPUT_SIZES.items()}
    return {'route': 'synthetic', 'frames': [{}] * n, 'ns': np.arange(n), 'partitions': partitions, 'cuts': set(),
            'cells': [(i, 0) for i in range(n)], 'speed': brain_targets.speed_class(np.asarray(kph)),
            'speed_weight': np.ones(n, dtype=np.float32), 'inputs': inputs, 'own_kph': np.asarray(kph), 'marks': [],
            'works': np.zeros(n, dtype=np.float32), 'works_weight': np.zeros(n, dtype=np.float32), 'reader': {}}


class SituationTest(unittest.TestCase):
    def test_baseline_can_include_every_train_tick_without_changing_replay_default(self):
        sample = drive(['train'] * 3 + ['test'], [20, 80, 80, 140])
        sample['measured'] = np.array([True, False, False, True])
        self.assertEqual(brain_replay.situation_baseline([sample]), {None: 20.0})
        self.assertEqual(brain_replay.situation_baseline([sample], all_ticks=True), {None: 80.0})

    def test_balancing_clips_then_preserves_the_total_speed_weight_across_drives(self):
        common = drive(['train'] * 40 + ['validation'], [50] * 41)
        common['measured'] = np.zeros(41, dtype=bool)  # all train ticks count, not just measured ticks
        common['speed_weight'][0] = 2
        rare = drive(['train'] * 5 + ['test'], [50] * 6)
        rare['speed_weight'][4] = 0
        rare['inputs']['lead'][0, 2] = 1
        rare['road_type'] = np.array([-1, 0, -1, 1, 1, -1])
        rare['inputs']['signs'][2, :2] = (0.25, 1)
        weights = brain_train.situation_weights([common, rare])
        normal = 45 / (41 * .25 + 4 * 4)
        np.testing.assert_allclose(weights[0], [normal * .25] * 40 + [1])
        np.testing.assert_allclose(weights[1], [normal * 4] * 4 + [1, 1])
        self.assertAlmostEqual(sum(float(np.dot(brain_targets.loss_weights(d, 'train')[0], w))
                                   for d, w in zip((common, rare), weights)), 45, places=5)
        self.assertEqual(rare['speed_weight'].tolist(), [1, 1, 1, 1, 0, 1])

    def test_training_swap_changes_only_matching_known_slots_in_the_passage(self):
        sample = drive(['train'] * 4, [50] * 4)
        inputs = sample['inputs']
        scale = brain_targets.BRAIN_INPUTS.SIGN_KPH_SCALE
        inputs['signs'][:, :2] = (50 / scale, 1)
        inputs['map_limit'][:, :] = (50 / scale, 1)
        rows = inputs['sign_slots']
        slots = brain_targets._slot_rows(rows)
        slots[:, :5, 0] = 1
        slots[:, :5, 6:8] = (50 / scale, 1)
        slots[:, 1, 5] = 1  # another kind
        slots[:, 2, 7] = 0  # unread
        slots[:, 3, 8] = 1  # end of limit
        slots[:, 4, 6] = 70 / scale
        rows[:, -2:] = .6
        groups = ('signs', 'sign_slots', 'map_limit')
        original = rows.copy()
        changed = brain_targets.swapped(sample, [np.array([1, 2])], groups)
        expected = original.copy()
        brain_targets._slot_rows(expected)[1:3, 0, 6] = 60 / scale
        np.testing.assert_array_equal(rows, original)
        np.testing.assert_array_equal(changed['inputs']['sign_slots'], expected)
        np.testing.assert_array_equal(changed['extra'], brain_targets.pack_inputs(changed['inputs'], groups))
        self.assertEqual(brain_targets.class_kph(changed['speed']).tolist(), [50, 60, 60, 50])

    def test_only_the_speed_loss_gets_the_situation_factor(self):
        import torch
        from nnslr_tools.brain_net import build_brain_net

        sample = drive(['train'], [50])
        sample['works_weight'][0] = .75
        sample['situation_weight'] = np.array([2], dtype=np.float32)
        groups = ('signs',)
        sample['extra'] = brain_targets.pack_inputs(sample['inputs'], groups)
        model = build_brain_net(groups, picture=False, pretrained=False, features=4, hidden=5)

        def loss(_logits, _works_logit, _speed, speed_weight, _works, works_weight, **_options):
            self.assertEqual(speed_weight.item(), 2)
            self.assertEqual(works_weight.item(), .75)
            raise ValueError('stop before optimisation')

        with patch.object(brain_train, 'brain_loss', side_effect=loss), self.assertRaisesRegex(ValueError, 'stop before'):
            brain_train.train_epoch(model, None, [sample], None, chunk=1, rng=random.Random(0), device=torch.device('cpu'))
        self.assertEqual(brain_targets.loss_weights(sample, 'train')[0][0], 1)

    def test_picture_off_bypasses_encoder_and_round_trips_in_the_checkpoint(self):
        import torch
        from nnslr_tools import brain_export, brain_net

        groups = ('signs',)
        model = brain_net.build_brain_net(groups, picture=False, pretrained=False, features=4, hidden=5, works=False).eval()
        images = torch.ones(1, 3, 3, 8, 8)
        extra = torch.zeros(1, 3, brain_targets.INPUT_SIZES['signs'])
        with patch.object(model.backbone, 'forward', side_effect=AssertionError), torch.no_grad():
            self.assertTrue(torch.equal(model.encode(images[0]), torch.zeros(3, 4)))
            self.assertTrue(torch.equal(model(images, extra)[0], model(-images, extra)[0]))
        self.assertEqual(model.gru.input_size, 4 + extra.shape[-1])
        saved = {'groups': groups, 'pool': model.pool, 'features': 4, 'hidden': 5, 'weights': model.state_dict()}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'brain-last.pt'
            torch.save({**saved, 'picture': False, 'picture_kind': 'brain'}, path)
            loaded, checkpoint = brain_export.load_brain(path)
            self.assertFalse(loaded.picture)
            self.assertEqual(brain_net.picture_kind_of(checkpoint), 'brain')
            with self.assertRaisesRegex(ValueError, 'offline comparison'):
                brain_export.export_brain_onnx(path, path.with_suffix('.onnx'))
            self.assertFalse(path.with_suffix('.onnx').exists())

            for mode in ({}, {'picture': 'brain'}, {'picture': True, 'picture_kind': 'brain'}):
                torch.save({**saved, **mode}, path)
                self.assertTrue(brain_export.load_brain(path)[0].picture)
            self.assertEqual(brain_net.picture_kind_of({'picture': 'brain'}), 'brain')
            self.assertEqual(brain_net.picture_kind_of({}), 'road_type')

    def test_final_results_and_legacy_checkpoint_are_for_the_kept_epoch(self):
        from nnslr_tools.model_roles import brain_digest

        for keep, epoch, error, test in (('best', 1, 4, 5), ('last', 2, 8, 15)):
            with self.subTest(keep=keep):
                found, standard, best, weights_seen = self.run_stubbed_epochs(keep)
                self.assertEqual([row['validation_error_by_place_kph'] for row in found['history']], [6, 4, 8])
                self.assertEqual([row['validation']['mean_error_kph'] for row in found['history']], [3.5, 4, 8])
                self.assertEqual(found['kept'], {'epoch': epoch, 'validation_error_by_place_kph': error})
                self.assertEqual(standard['epoch'], epoch)
                self.assertEqual(best['epoch'], 1)
                self.assertEqual(standard['weights']['speed_head.bias'][0].item(), epoch)
                self.assertEqual(best['weights']['speed_head.bias'][0].item(), 1)
                self.assertIs(standard['picture'], False)
                self.assertIs(found['picture'], False)
                self.assertIs(found['balance'], True)
                self.assertEqual(found['test']['speed']['mean_error_kph'], test)
                self.assertEqual(found['test_cold']['speed']['mean_error_kph'], test)
                self.assertEqual(weights_seen, [epoch])
                test_digest = brain_digest(found)['situation']['test']
                self.assertEqual(test_digest['no_sign']['mean_error_kph'], test)
                self.assertEqual(test_digest['no_sign']['baseline_mean_error_kph'], 30)
                self.assertEqual(test_digest['mean_error_by_place_kph'], test)

    def run_stubbed_epochs(self, keep, validation=True):
        import torch
        from nnslr_tools.brain_net import build_brain_net

        sample = drive(['train'] * 3 + ['validation' if validation else 'train'] * 4 + ['test'] * 2,
                       [20, 80, 80, 50, 50, 50, 50, 50, 50])
        sample['measured'] = np.array([True, False, False, True, True, True, True, True, True])
        sample['cells'][3:6] = [(0, 0)] * 3
        device = 'cpu'
        groups = brain_train.learned_inputs()
        model = build_brain_net(groups, picture=False, pretrained=False, features=4, hidden=5, works=False)
        epochs = iter(range(3))
        speeds = [[51, 51, 51, 61, 51, 51], [46, 46, 46, 46, 45, 45], [58, 58, 58, 58, 65, 65]]
        weights_seen = []

        def step(net, _optimizer, drives, *_args, **_kwargs):
            self.assertEqual(len(drives), 1)
            self.assertTrue((drives[0]['situation_weight'] == 1).all())
            with torch.no_grad():
                net.speed_head.bias.fill_(next(epochs))
            return 0.0

        def predict(net, _sample, store, *, cold=False, **_options):
            self.assertIsNone(store)
            epoch = int(net.speed_head.bias[0].item())
            if cold:
                weights_seen.append(epoch)
            return {'kph': np.array([20, 80, 80, *speeds[epoch]]), 'works': None}

        with tempfile.TemporaryDirectory() as folder:
            root, run = Path(folder), Path(folder) / 'run'
            with patch.multiple(brain_train, dataset=lambda *_a, **_kw: [sample],
                                training_counts=lambda _root: {'drives': 1, 'hours': 1, 'sunnypilot_on': 0},
                                check_drives=lambda _drives: None, read_split=lambda _root: {'sha256': 'split'},
                                build_brain_net=lambda *_a, **_kw: model, predict=predict, train_epoch=step,
                                epoch_drives=lambda drives, *_a: drives, FEATURES=4, HIDDEN=5):
                result = brain_train.train_brain(root, run, None, epochs=3, device=device, works=False, types=False,
                                                 road=False, applicability=False, picture=False, balance=True,
                                                 keep=keep, log=lambda _: None)
            self.assertEqual(json.loads((run / 'result.json').read_text()), result)
            return (result, torch.load(run / brain_train.CHECKPOINT, weights_only=False),
                    torch.load(run / brain_train.BEST_CHECKPOINT, weights_only=False), weights_seen)

    def test_no_measured_validation_falls_back_to_last(self):
        result, standard, best, _ = self.run_stubbed_epochs('best', validation=False)
        self.assertEqual(result['kept'], {'epoch': 2, 'validation_error_by_place_kph': None})
        self.assertEqual(result['situation']['validation'], {'ticks': 0})
        self.assertEqual((standard['epoch'], best['epoch']), (2, 2))


if __name__ == '__main__':
    unittest.main()
