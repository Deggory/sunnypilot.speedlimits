"""Paired replay on synthetic inputs and a tiny CPU brain; no recorded drives or weights."""
import unittest
from unittest.mock import patch


class BrainReplayTest(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from nnslr_tools import brain_replay, brain_targets

        self.np, self.replay, self.targets = np, brain_replay, brain_targets

    def drive(self, groups=('signs', 'sign_slots', 'map_limit'), count=32, confirmation=6):
        np, targets = self.np, self.targets
        scale = targets.BRAIN_INPUTS.SIGN_KPH_SCALE
        inputs = {name: np.zeros((count, size), dtype=np.float32) for name, size in targets.INPUT_SIZES.items()}
        signs = inputs['signs']
        signs[:confirmation, :2] = (80 / scale, 1)
        signs[confirmation:, :2] = (50 / scale, 1)
        inputs['map_limit'][:] = (50 / scale, 1)
        inputs['map_limit'][confirmation + 2] = (70 / scale, 1)
        inputs['map_limit'][confirmation + 3, 1] = 0
        road_type = np.full(count, self.replay.ROAD_TYPES.index('rural'))
        return {'route': 'synthetic', 'frames': [{}] * count, 'cuts': set(), 'partitions': ['validation'] * count,
                'road_type': road_type, 'inputs': inputs, 'extra': targets.pack_inputs(inputs, groups),
                'speed': targets.speed_class(signs[:, 0] * scale), 'speed_weight': np.ones(count),
                'works_weight': np.zeros(count), 'own_kph': np.full(count, 50.0)}

    def model(self, groups, **options):
        import torch
        from nnslr_tools.brain_net import build_brain_net

        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(17)
            return build_brain_net(groups, pretrained=False, pool='mean', works=False, features=4, hidden=5,
                                   kph=(40, 45, 50, 55, 60, 80), number_path='bell', **options)

    @staticmethod
    def store():
        class ZeroImages:
            def batch(self, requests):
                import torch

                [(_route, ticks)] = requests
                return torch.zeros(1, len(ticks), 3, 32, 32, device='cpu')
        return ZeroImages()

    def test_swap_changes_only_the_matching_readings_and_answers(self):
        np, targets = self.np, self.targets
        groups = ('signs', 'sign_slots', 'map_limit', 'lead', 'road_signs')
        drive = self.drive(groups)
        inputs, confirm = drive['inputs'], 6
        scale = targets.BRAIN_INPUTS.SIGN_KPH_SCALE
        signs = inputs['signs']
        signs[3:12, 8:10] = (50 / scale, 1)
        signs[7, 9] = 0
        signs[8, 8] = 70 / scale
        signs[27, 6] = 1
        slots = self.replay._slot_rows(inputs['sign_slots'])
        slots[2:28, :6, 0] = 1
        slots[2:28, :6, 6:8] = (50 / scale, 1)
        slots[:, 0, 3:5] = (.12, .18)
        slots[:, 1, 6] = 70 / scale
        slots[:, 2, 5] = 1  # Other kind of sign.
        slots[:, 3, 7] = 0  # Unknown.
        slots[:, 4, 8] = 1  # End of limit.
        slots[:, 5, 0] = 0  # Empty.
        drive['speed'][6:10] = targets.speed_class(np.array([52, 35, np.nan, 50]))
        inputs['road_signs'][:, -4:] = 1
        sizes = {**targets.INPUT_SIZES, 'road_signs': targets.INPUT_SIZES['road_signs'] - 4}
        drive['extra'] = targets.pack_inputs(inputs, groups, sizes)
        original = {name: array.copy() for name, array in inputs.items()}
        speed, extra = drive['speed'].copy(), drive['extra'].copy()

        swapped = self.replay.coherent_swap(drive, np.arange(confirm, 32), groups)

        for name in inputs:
            np.testing.assert_array_equal(inputs[name], original[name])
        np.testing.assert_array_equal(drive['speed'], speed)
        np.testing.assert_array_equal(drive['extra'], extra)
        expected = original['signs'].copy()
        expected[6:27, 0] = 60 / scale
        expected[[3, 4, 5, 6, 9, 10, 11], 8] = 60 / scale
        np.testing.assert_array_equal(swapped['inputs']['signs'], expected)
        expected = original['sign_slots'].copy()
        self.replay._slot_rows(expected)[2:27, 0, 6] = 60 / scale
        np.testing.assert_array_equal(swapped['inputs']['sign_slots'], expected)
        expected = original['map_limit'].copy()
        expected[np.r_[6:8, 10:27], 0] = 60 / scale
        np.testing.assert_array_equal(swapped['inputs']['map_limit'], expected)
        self.assertEqual(targets.class_kph(swapped['speed'][5:11]).tolist(), [80, 62, 35, 0, 60, 60])
        np.testing.assert_array_equal(swapped['extra'], targets.pack_inputs(swapped['inputs'], groups, sizes))
        for name in inputs.keys() - {'signs', 'sign_slots', 'map_limit'}:
            np.testing.assert_array_equal(swapped['inputs'][name], original[name])
        self.assertEqual(self.replay._passage(drive, np.arange(6, 32)), (2, 6, 27, 50))

    def test_decode_peaks_and_first_class_at_half(self):
        np = self.np
        found = self.replay.decode(np.array([[.51, 0, .49], [.49, 0, .51], [.5, 0, .5], [.4, .2, .4]]),
                                   np.array([30, 45, 70]))
        np.testing.assert_array_equal(found['argmax'], [30, 70, 30, 30])
        np.testing.assert_array_equal(found['median'], [30, 70, 30, 45])
        np.testing.assert_allclose(found['mean'], [49.6, 50.4, 50, 49])

    def test_passage_does_not_use_a_stale_reading_from_an_earlier_attempt(self):
        np, replay = self.np, self.replay
        drive = self.drive(confirmation=16)
        signs = drive['inputs']['signs']
        signs[2:, 8:10] = (50 / replay.SCALE, 1)
        signs[2:15, 10] = np.log1p(np.arange(13) / replay.RATE_HZ) / replay.BRAIN_INPUTS.SIGN_AGE_SCALE
        self.assertEqual(replay._passage(drive, np.arange(16, 32)), (15, 16, 32, 50))

    def test_probabilities_match_predict_with_cuts_and_keep_map_in_ablation(self):
        import torch
        from nnslr_tools.brain_train import predict

        np, replay = self.np, self.replay
        groups = ('signs', 'map_limit')
        drive = self.drive(groups, count=18)
        drive['cuts'] = {7, 12}
        model = self.model(groups, map_path=True)
        with torch.no_grad():
            model.trust.bias.fill_(1)
            model.trust_map.bias.fill_(.8)
        store = self.store()
        actual = replay.speed_probabilities(model, drive, store, chunk=5, device='cpu')
        expected = predict(model, drive, store, chunk=5, device='cpu')
        np.testing.assert_array_equal(replay.decode(actual['probs'], model.class_kph.numpy())['argmax'], expected['kph'])
        np.testing.assert_allclose(actual['without_sign'].sum(axis=1), 1, atol=1e-6)
        np.testing.assert_allclose(actual['trust'], 1)
        self.assertGreater(float(np.max(np.abs(actual['probs'] - actual['without_sign']))), .05)
        # Zero only the head term independently, leaving input to the GRU untouched.
        with torch.no_grad():
            model.trust.bias.zero_()
        bare = replay.speed_probabilities(model, drive, store, chunk=5, device='cpu')
        np.testing.assert_allclose(actual['without_sign'], bare['probs'], atol=1e-6)
        with torch.no_grad():
            model.trust_map.bias.zero_()
        no_map = replay.speed_probabilities(model, drive, store, chunk=5, device='cpu')
        self.assertGreater(float(np.max(np.abs(no_map['probs'] - bare['probs']))), .05)

    def test_pair_shares_a_state_computed_with_cuts_and_stops_at_the_next_cut(self):
        import torch

        np, replay = self.np, self.replay
        # The original and the swap give exactly the same model inputs when it learns only the lead.
        groups = ('lead',)
        drive = self.drive(groups)
        drive['cuts'] = {1, 24}
        drive['inputs']['signs'][4:8, 8:10] = (50 / replay.SCALE, 1)
        model, store = self.model(groups).eval(), self.store()
        with torch.no_grad():
            _, _, expected = model(store.batch([('synthetic', [1, 2, 3])]), torch.from_numpy(drive['extra'][1:4])[None])
        states = []
        hook = model.register_forward_pre_hook(lambda _model, args: states.append(args[2]))
        self.addCleanup(hook.remove)
        result = replay.paired_replay(model, drive, np.arange(6, 32), store, groups, device='cpu')
        self.assertEqual(result['first_reading_tick'], 4)
        self.assertEqual(result['confirmation_tick'], 6)
        self.assertEqual(result['ticks'].tolist(), list(range(4, 24)))
        self.assertIsNone(states[1])
        self.assertIsNot(states[-2], states[-1])
        torch.testing.assert_close(states[-2], expected)
        torch.testing.assert_close(states[-1], expected)
        for name in replay.DECODERS:
            np.testing.assert_array_equal(result['original'][name], result['swapped'][name])
        ratio = replay.response_ratio(result, drive, 'validation')
        self.assertEqual(ratio['direction'], 'up')
        self.assertEqual(ratio['windows']['2-5s']['median_r'], dict.fromkeys(replay.DECODERS, 0.0))
        result['swapped_limit'] = result['original_limit']
        ratio = replay.response_ratio(result, drive, 'validation')
        self.assertEqual(ratio['windows']['2-5s']['median_r'], dict.fromkeys(replay.DECODERS))

    def test_state_from_one_pass_equals_running_up_to_the_first_reading(self):
        np, replay = self.np, self.replay
        groups = ('signs', 'sign_slots')
        drive = self.drive(groups)
        drive['cuts'] = {2}
        drive['inputs']['signs'][4:8, 8:10] = (50 / replay.SCALE, 1)
        model, store = self.model(groups).eval(), self.store()
        whole = replay.speed_probabilities(model, drive, store, chunk=3, device='cpu', states_at=[4])
        plain = replay.speed_probabilities(model, drive, store, chunk=3, device='cpu')
        np.testing.assert_allclose(whole['probs'], plain['probs'], rtol=1e-5, atol=1e-6)
        given = replay.paired_replay(model, drive, np.arange(6, 32), store, groups, device='cpu', state=whole['states'][4])
        alone = replay.paired_replay(model, drive, np.arange(6, 32), store, groups, device='cpu')
        for branch in ('original', 'swapped'):
            for name in replay.DECODERS:
                np.testing.assert_allclose(given[branch][name], alone[branch][name], rtol=1e-5, atol=1e-5)

    def test_situation_splits_by_number_state_road_type_and_lead_against_the_train_baseline(self):
        np, replay, targets = self.np, self.replay, self.targets
        scale = targets.BRAIN_INPUTS.SIGN_KPH_SCALE
        train = self.drive()
        train['partitions'] = ['train'] * 32
        train['speed'][:] = targets.speed_class(np.full(32, 46.0))
        train['inputs']['signs'][:6, 1] = 0  # no number yet on the first ticks
        train['speed'][:6] = targets.speed_class(np.full(6, 70.0))
        train['cells'] = [(1, 1)] * 32
        baseline = replay.situation_baseline([train])
        self.assertEqual(baseline, {None: 70.0, 50: 46.0})
        drive = self.drive()
        drive['cells'] = [(2, 2)] * 16 + [(3, 3)] * 16
        drive['road_type'][20:] = replay.ROAD_TYPES.index('ramp')
        drive['speed'][20:] = targets.speed_class(np.full(12, 30.0))
        drive['inputs']['lead'][24:, 2] = 1
        said = np.full(32, 50.0)
        said[20:] = 40.0
        with patch.object(replay, 'SITUATION_MIN_TICKS', 10):
            found = replay.situation_metrics([drive], [said], 'validation', baseline)
        self.assertEqual(found['all']['ticks'], 32)
        self.assertEqual(found['all']['places'], 2)
        self.assertEqual(set(found['by']['state']), {'number, fresh'})
        self.assertEqual(found['by']['type']['ramp']['ticks'], 12)
        self.assertAlmostEqual(found['by']['type']['ramp']['mean_error_kph'], 10.0)
        self.assertAlmostEqual(found['by']['type']['ramp']['baseline_mean_error_kph'], 16.0)
        self.assertEqual(found['by']['lead']['lead car']['ticks'], 8)
        ramp = next(group for group in found['within_number']['groups'] if group['type'] == 'ramp')
        self.assertLess(ramp['brain_kph'], 0)
        self.assertLess(ramp['driver_kph'], ramp['brain_kph'])
        self.assertIsNotNone(found['within_number']['slope'])
        old = drive['inputs']['signs']
        old[:, 5] = np.log1p(120) / targets.BRAIN_INPUTS.SIGN_AGE_SCALE
        old[:3, 1], old[3:6, 6] = 0, 1
        states = replay._number_states(drive)
        self.assertEqual(states[:6].tolist(), ['none yet'] * 3 + ['after an end'] * 3)
        self.assertEqual(states[6], 'number, old')

    def test_situation_adds_the_type_baseline_the_time_after_an_end_and_the_entry_profile(self):
        np, replay, targets = self.np, self.replay, self.targets
        train = self.drive()
        train['partitions'] = ['train'] * 32
        train['speed'][:] = targets.speed_class(np.full(32, 46.0))
        train['road_type'][20:] = replay.ROAD_TYPES.index('ramp')
        train['speed'][20:] = targets.speed_class(np.full(12, 34.0))
        train['cells'] = [(1, 1)] * 32
        baseline, typed = replay.situation_baseline([train]), replay.situation_type_baseline([train])
        self.assertEqual(typed, {'80|rural': 46.0, '50|rural': 46.0, '50|ramp': 34.0})
        drive = self.drive()
        drive['cells'] = [(2, 2)] * 32
        drive['road_type'][20:] = replay.ROAD_TYPES.index('ramp')
        drive['speed'][20:] = targets.speed_class(np.full(12, 30.0))
        drive['inputs']['signs'][8:16, 6] = 1  # an end from tick 8, at 2 s, for 2 s
        said = np.full(32, 50.0)
        said[24:] = 40.0  # slows one second late
        found = replay.situation_metrics([drive], [said], 'validation', baseline, typed)
        ramp = found['by']['type']['ramp']
        self.assertAlmostEqual(ramp['baseline_mean_error_kph'], 16.0)
        self.assertAlmostEqual(ramp['type_baseline_mean_error_kph'], 4.0)
        self.assertEqual(found['after_end']['0-10s']['ticks'], 8)
        self.assertEqual(found['after_end']['10-30s']['ticks'], 0)
        np.testing.assert_allclose(replay._seconds_since_end(drive)[8:16], np.arange(8) / 4)
        entry = found['entries']['ramp']
        self.assertEqual(entry['entries'], 1)
        self.assertEqual(entry['phases']['before']['ticks'], 20)
        self.assertAlmostEqual(entry['phases']['first 10 s']['brain_minus_driver_kph'], (4 * 20 + 8 * 10) / 12)
        first = next(row for row in entry['profile'] if row['second'] == 0)
        self.assertEqual((first['driver_kph'], first['brain_kph']), (30.0, 50.0))
        self.assertNotIn('merge', found['entries'])

    def test_pair_follows_a_strong_sign_path_in_both_directions(self):
        import torch

        np, replay = self.np, self.replay
        groups = ('signs', 'sign_slots')
        drive = self.drive(groups)
        model, store = self.model(groups), self.store()
        with torch.no_grad():
            model.trust.bias.fill_(5)
        for number, alternate, direction in [(50, 60, 'up'), (60, 50, 'down')]:
            with self.subTest(direction=direction):
                drive['inputs']['signs'][4:8, 8:10] = (number / replay.SCALE, 1)
                drive['inputs']['signs'][6:, 0] = number / replay.SCALE
                drive['speed'][6:] = self.targets.speed_class(number)
                drive['extra'] = self.targets.pack_inputs(drive['inputs'], groups)
                drive['cuts'] = {4}
                result = replay.paired_replay(model, drive, np.arange(6, 32), store, groups, device='cpu')
                self.assertEqual(result['direction'], direction)
                self.assertEqual(result['original']['argmax'][:3].tolist(), [80, 80, number])
                self.assertEqual(result['swapped']['argmax'][:3].tolist(), [80, 80, alternate])
                windows = replay.response_ratio(result, drive, 'validation')['windows']
                self.assertEqual(windows['2-5s']['median_r']['argmax'], 1.0)

    def test_response_windows_and_stretch_labels_ignore_non_said_ticks_for_free_road(self):
        np, replay = self.np, self.replay
        drive = self.drive()
        original, alternative = np.full(26, 50), np.full(26, 60)
        faster = np.r_[np.full(4, 54), np.full(4, 60), np.full(12, 57), np.full(6, 60)]
        result = {'ticks': np.arange(6, 32), 'confirmation_tick': 6, 'direction': 'up', 'original_limit': original,
                  'swapped_limit': alternative, 'original': dict.fromkeys(replay.DECODERS, original),
                  'swapped': dict.fromkeys(replay.DECODERS, faster)}
        drive['partitions'][22:] = ['test'] * 10
        drive['speed'][14] = self.targets.speed_class(30)
        drive['inputs']['lead'][6, 2] = 1
        drive['inputs']['lead'][14, 2] = 1
        drive['road_type'][14] = replay.RAMP
        windows = replay.response_ratio(result, drive, 'validation')['windows']
        self.assertEqual(windows['0-1s']['kind'], 'lead car')
        self.assertEqual(windows['0-1s']['median_r']['argmax'], .4)
        self.assertEqual(windows['2-5s']['kind'], 'free road')
        self.assertEqual(windows['2-5s']['frames'], 7)
        self.assertEqual(windows['2-5s']['median_r']['argmax'], .7)
        drive['road_type'][15] = replay.RAMP
        self.assertEqual(replay.response_ratio(result, drive, 'validation')['windows']['2-5s']['kind'], 'ramp')
        drive['partitions'][6] = 'test'
        self.assertEqual(replay.response_ratio(result, drive, 'validation')['windows']['2-5s']['frames'], 0)

    def test_decoder_metrics_count_measured_errors_and_steady_pairs_and_require_a_full_second(self):
        np, replay = self.np, self.replay
        drive = self.drive()
        drive['measured'] = np.arange(32) % 4 == 0
        drive['partitions'][28:] = ['test'] * 4
        drive['cuts'] = {23}
        drive['inputs']['signs'][18:, 0] = 60 / replay.SCALE
        drive['speed'][18:] = self.targets.speed_class(60)
        original = np.array([80, 80, 80, 80, 80, 80, 50, 50, 50, 53, 50, 50, 50, 50, 50, 50,
                             50, 50, 50, 50, 60, 60, 60, 60, 60, 60, 60, 60, 199, 199, 199, 199])
        better = np.r_[np.full(6, 80), np.full(12, 50), np.full(14, 60)]
        found = replay.decoder_metrics([drive], [{'argmax': original, 'mean': better, 'median': better}], 'validation')
        self.assertEqual(found['argmax']['frames'], 7)
        self.assertEqual(found['argmax']['right'], 1)
        self.assertEqual(found['argmax']['mean_error_kph'], 0)
        self.assertGreater(found['argmax']['p95_jump_kph'], 0)
        self.assertEqual(found['argmax']['changes'], 2)
        self.assertEqual(found['argmax']['never_reached'], 1)
        self.assertEqual(found['argmax']['delay_s'], 1)
        self.assertEqual(found['mean']['never_reached'], 0)
        self.assertEqual(found['mean']['delay_s'], 0)

    def test_bootstrap_samples_drives_instead_of_independent_passages(self):
        rows = {'a': [0] * 30, 'b': [.9] * 2, 'empty': [None]}
        found = self.replay.bootstrap_by_drive(rows)
        self.assertEqual(found, {'median': 0, 'ci95': (0, .9), 'drives': 2, 'passages': 32})
        self.assertEqual(self.replay.bootstrap_by_drive(rows), found)
        self.assertIsNone(self.replay.bootstrap_by_drive([])['ci95'])

    def test_decision_uses_only_late_said_validation_free_road_and_guards_the_decoder(self):
        def row(route, direction, ratio, *, partition='validation', kind='free road', said=2):
            window = {'kind': kind, 'said_frames': said, 'median_r': {'argmax': ratio}}
            early = {**window, 'median_r': {'argmax': 2.0}}
            return {'route': route, 'direction': direction, 'partition': partition, 'windows': {'0-1s': early, '2-5s': window}}

        decide = self.replay.decision
        baseline = {'p95_jump_kph': 3, 'mean_error_kph': 1, 'right': .9, 'delay_s': .75, 'never_reached': 2}
        mean = {**baseline, 'p95_jump_kph': 1, 'mean_error_kph': 1.5, 'right': .89, 'delay_s': 1.25}
        median = {**mean, 'p95_jump_kph': .5, 'never_reached': 3}
        metrics = {'validation': {'argmax': baseline, 'mean': mean, 'median': median}}
        rows = [row('a', 'up', .5), row('b', 'down', .6)]
        result = decide({'responses': rows, 'decoders': metrics})
        self.assertEqual(result['training'], 'no training needed for this')
        self.assertEqual(result['decoder'], 'mean')
        self.assertEqual(decide({'responses': [row('a', 'up', .4)]})['training'],
                         'train: coherent swap, fresh init, best checkpoint')
        rows = [row('a', 'up', .0), row('b', 'up', .8), row('c', 'down', .7, partition='test'),
                row('d', 'down', .8, kind='ramp'), row('e', 'down', .8, said=0)]
        result = decide({'responses': rows})
        self.assertEqual(result['training'], 'need more passages')
        self.assertEqual(result['decoder'], 'argmax')
        self.assertIsNone(result['validation_response']['down']['median'])
        for key, bad in [('mean_error_kph', 1.51), ('right', .889), ('delay_s', 1.26), ('p95_jump_kph', 3)]:
            with self.subTest(metric=key), patch.dict(mean, {key: bad}):
                self.assertEqual(decide({'decoders': metrics})['decoder'], 'argmax')
        with patch.dict(median, {'never_reached': 2}):
            self.assertEqual(decide({'decoders': metrics})['decoder'], 'median')
