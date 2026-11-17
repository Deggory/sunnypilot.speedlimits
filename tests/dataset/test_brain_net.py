"""The last step from a label to the weights: which frames give a gradient, and where it goes. These need torch, which only
.venv-gpu has, and .venv-gpu has no pytest: they are skipped in .venv. Small pictures on the CPU, no training:
uv pip install --target /tmp/pytest-gpu --python .venv-gpu/bin/python pytest
PYTHONPATH=/tmp/pytest-gpu .venv-gpu/bin/python -m pytest -o addopts="" -p no:cacheprovider tests/dataset/test_brain_net.py"""
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')

GROUPS = ('signs', 'plate', 'hour', 'lead', 'country')
SIZE = 0
T = 5


def setup_module():
    global SIZE
    from nnslr_tools.brain_targets import INPUT_SIZES
    SIZE = sum(INPUT_SIZES[name] for name in GROUPS)


def net(**options):
    from nnslr_tools.brain_net import build_brain_net
    torch.manual_seed(0)
    model = build_brain_net(GROUPS, pretrained=False, **options)
    model.train()
    for layer in model.modules():
        if isinstance(layer, torch.nn.Dropout):
            layer.p = 0.0
    return model


def batch(t=T):
    torch.manual_seed(1)
    return torch.rand(1, t, 3, 144, 256), torch.rand(1, t, SIZE)


def answers(t=T, speed_on=(), works_on=()):
    from nnslr_tools.brain_targets import speed_class
    speed = torch.full((1, t), speed_class(100), dtype=torch.long)
    speed_weight, works_weight = torch.zeros(1, t), torch.zeros(1, t)
    speed_weight[0, list(speed_on)] = 1
    works_weight[0, list(works_on)] = 1
    return speed, speed_weight, torch.ones(1, t), works_weight


def gradient_of(model, part):
    found = [value.grad for name, value in model.named_parameters() if name.startswith(part) and value.grad is not None]
    return sum(float(grad.abs().sum()) for grad in found)


def backward(model, images, extra, labels):
    from nnslr_tools.brain_net import brain_loss
    model.zero_grad()
    speed_logits, works_logit, _state = model(images, extra)
    speed, speed_weight, works, works_weight = labels
    loss = brain_loss(speed_logits, works_logit, speed, speed_weight, works, works_weight)
    loss.backward()
    return loss


def test_the_network_says_a_speed_and_works_on_every_frame():
    from nnslr_tools.brain_targets import SPEED_CLASSES
    speed, works, state = net()(*batch())
    assert speed.shape == (1, T, SPEED_CLASSES)
    assert works.shape == (1, T)
    assert state.shape == (1, 128)


def test_the_table_of_today_is_a_class_a_km_h_with_none_ahead_after_it():
    # The heads of the limit ahead are built only for a checkpoint that has them (7 to 11 Nov 2026).
    from nnslr_tools.brain_targets import SPEED_CLASSES
    model = net(ahead=True)
    assert SPEED_CLASSES == 200
    assert model.class_kph.tolist() == [float(kph) for kph in range(1, 201)]
    assert (model.speed_head.out_features, model.next_head.out_features) == (200, 201)
    assert model.near_kph == pytest.approx(0.5)


def test_a_brain_of_today_says_the_speed_and_has_no_limit_ahead():
    import inspect

    from nnslr_tools import brain_net
    model = net()
    assert not model.ahead
    assert not [name for name in model.state_dict() if name.startswith(('next_head.', 'trust_next.', 'metres_head.',
                                                                         'metres_gate.'))]
    # Nothing teaches a next limit or its metres: the loss takes no such answer.
    assert 'ahead' not in inspect.signature(brain_net.brain_loss).parameters
    assert not hasattr(brain_net, 'NEXT_LOSS') and not hasattr(brain_net, 'METRES_LOSS')
    # A checkpoint that has the heads still gets them, and its next limit and metres beside the speed.
    old = net(ahead=True)
    said = brain_net.outputs(old(*batch(), ahead=True), ahead=True)
    assert said['next'].shape == (1, T, 201) and said['metres'].shape == (1, T)


def test_a_brain_of_5_km_h_classes_keeps_its_table():
    model = net(ahead=True, kph=list(range(5, 205, 5)))
    assert model.class_kph.tolist() == [float(kph) for kph in range(5, 205, 5)]
    assert (model.speed_head.out_features, model.next_head.out_features) == (40, 41)
    assert model.near_kph == pytest.approx(2.5)
    trusting(model, 3.0)
    assert model(batch(2)[0], number_row(50))[0].argmax(dim=-1).tolist() == [[9, 9]]
    assert 'near_kph' not in model.state_dict()


@pytest.mark.parametrize('kph', [[], [5, 5, 10], [10, 5]])
def test_a_table_is_km_h_going_up(kph):
    from nnslr_tools.brain_net import build_brain_net
    with pytest.raises(ValueError):
        build_brain_net(GROUPS, pretrained=False, kph=kph)


def test_his_speed_is_a_bell_around_its_class():
    import math

    from nnslr_tools.brain_net import SPEED_SIGMA_KPH, speed_targets
    from nnslr_tools.brain_targets import speed_class
    targets = speed_targets(torch.tensor([speed_class(52), speed_class(1), speed_class(200)]))
    assert targets.shape == (3, 200)
    assert targets.sum(dim=-1).tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert targets.argmax(dim=-1).tolist() == [speed_class(52), 0, 199]
    near = targets[0, speed_class(54)] / targets[0, speed_class(52)]
    assert float(near) == pytest.approx(math.exp(-0.5 * (2 / SPEED_SIGMA_KPH) ** 2))
    assert float(targets[0, speed_class(50)]) == pytest.approx(float(targets[0, speed_class(54)]))
    assert float(targets[0, speed_class(120)]) < 1e-12


def test_on_the_table_of_before_the_bell_is_almost_the_class_alone():
    from nnslr_tools.brain_net import speed_targets
    targets = speed_targets(torch.tensor([9]), kph=torch.arange(5, 205, 5, dtype=torch.float32))
    assert targets.shape == (1, 40)
    assert float(targets[0, 9]) > 0.85
    assert float(targets[0, 8]) == pytest.approx(float(targets[0, 10]))


def test_a_near_speed_costs_less_than_a_far_one():
    from nnslr_tools.brain_net import brain_loss
    from nnslr_tools.brain_targets import speed_class
    speed, speed_weight, works, works_weight = answers(1, speed_on=[0])
    speed[0, 0] = speed_class(52)

    def said(kph):
        logits = torch.zeros(1, 1, 200)
        logits[0, 0, speed_class(kph)] = 10
        return float(brain_loss(logits, None, speed, speed_weight, works, works_weight))
    assert said(52) < said(53) < said(120)
    assert said(51) == pytest.approx(said(53))


def test_a_frame_without_a_class_and_no_weight_adds_nothing():
    from nnslr_tools.brain_net import brain_loss
    speed_logits, works_logit, _state = net()(*batch())
    speed, speed_weight, works, works_weight = answers(speed_on=[0, 1])
    without = speed.clone()
    without[0, 3] = -1
    assert float(brain_loss(speed_logits, None, without, speed_weight, works, works_weight)) == pytest.approx(
        float(brain_loss(speed_logits, None, speed, speed_weight, works, works_weight)))


@pytest.mark.parametrize('pool', ['mean', 'grid'])
def test_a_mark_reaches_the_memory_and_the_picture_features_and_not_the_speed_head(pool):
    model = net(pool=pool)
    backward(model, *batch(), answers(works_on=[1, 2]))
    assert gradient_of(model, 'works_head') > 0
    assert gradient_of(model, 'gru') > 0
    assert gradient_of(model, 'project') > 0
    assert gradient_of(model, 'speed_head') == 0
    assert gradient_of(model, 'trust') == 0


def test_one_step_on_a_mark_changes_what_the_network_says_of_the_speed():
    model = net()
    images, extra = batch()
    with torch.no_grad():
        before, _works, _state = model(images, extra)
    head = model.speed_head.weight.clone()
    backward(model, images, extra, answers(works_on=[1, 2]))
    torch.optim.SGD(model.parameters(), lr=0.1).step()
    with torch.no_grad():
        after, _works, _state = model(images, extra)
    assert torch.equal(head, model.speed_head.weight)
    assert not torch.equal(before, after)


def test_a_stretch_with_no_mark_gives_no_gradient_at_all():
    model = net()
    loss = backward(model, *batch(), answers())
    assert float(loss) == 0
    for part in ('works_head', 'speed_head', 'gru', 'project', 'backbone'):
        assert gradient_of(model, part) == 0, part


def test_a_frame_out_of_the_loss_counts_for_nothing_whatever_its_answer_says():
    from nnslr_tools.brain_net import brain_loss
    speed_logits, works_logit, _state = net()(*batch())
    speed, speed_weight, works, works_weight = answers(speed_on=[0], works_on=[0])
    other, other_works = speed.clone(), works.clone()
    other[0, 3], other_works[0, 3] = 3, 0
    assert float(brain_loss(speed_logits, works_logit, speed, speed_weight, works, works_weight)) == pytest.approx(
        float(brain_loss(speed_logits, works_logit, other, speed_weight, other_works, works_weight)))


def test_the_works_error_is_a_share_of_the_loss():
    from nnslr_tools.brain_net import WORKS_LOSS, brain_loss
    speed_logits, works_logit, _state = net()(*batch())
    speed, speed_weight, works, works_weight = answers(speed_on=[0, 1], works_on=[2])
    both = brain_loss(speed_logits, works_logit, speed, speed_weight, works, works_weight)
    alone = brain_loss(speed_logits, None, speed, speed_weight, works, works_weight)
    none = brain_loss(speed_logits, works_logit, speed, speed_weight, works, works_weight, works_share=0.0)
    assert float(none) == pytest.approx(float(alone))
    assert float(both) > float(alone)
    assert WORKS_LOSS == pytest.approx(0.3)


def test_with_the_frames_of_a_piece_an_answer_weighs_the_same_in_a_piece_with_few():
    from nnslr_tools.brain_net import brain_loss
    speed_logits, works_logit, _state = net()(*batch())
    speed, speed_weight, works, works_weight = answers(speed_on=[0, 1, 2, 3], works_on=[0])
    alone = speed_weight.clone()
    alone[0, 1:] = 0
    both = brain_loss(speed_logits, None, speed, speed_weight, works, works_weight, frames=T)
    first = brain_loss(speed_logits, None, speed, alone, works, works_weight, frames=T)
    assert float(first) < float(both)
    from nnslr_tools.brain_net import speed_targets
    errors = torch.nn.functional.cross_entropy(speed_logits[0], speed_targets(speed[0]), reduction='none')
    assert float(first) == pytest.approx(float(errors[0]) / T)


@pytest.mark.parametrize('pool', ['mean', 'grid'])
def test_a_frame_never_sees_the_frames_after_it_also_in_training(pool):
    model = net(pool=pool)
    images, extra = batch()
    changed, changed_extra = images.clone(), extra.clone()
    changed[0, 3:] = torch.rand_like(changed[0, 3:])
    changed_extra[0, 3:] = 0
    with torch.no_grad():
        speed, works, _state = model(images, extra)
        other, other_works, _state = model(changed, changed_extra)
    assert torch.equal(speed[0, :3], other[0, :3])
    assert torch.equal(works[0, :3], other_works[0, :3])
    assert not torch.equal(speed[0, 3:], other[0, 3:])


def test_a_training_step_leaves_the_encoder_as_it_is():
    model = net()
    before = {name: value.clone() for name, value in model.backbone.state_dict().items()}
    optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad], lr=1.0)
    backward(model, *batch(), answers(speed_on=[1, 2], works_on=[1, 2]))
    optimizer.step()
    assert gradient_of(model, 'backbone') == 0
    assert all(torch.equal(value, before[name]) for name, value in model.backbone.state_dict().items())


def test_the_scale_of_the_picture_does_not_reach_the_memory():
    model = net()
    model.eval()
    images, extra = batch()
    with torch.no_grad():
        before, _works, _state = model(images, extra)
        model.project[0].weight.mul_(1000)
        model.project[0].bias.mul_(1000)
        after, _works, _state = model(images, extra)
    assert float((before - after).abs().max()) < 1e-2


def test_training_does_not_move_the_batch_norm():
    model = net()
    before = {name: value.clone() for name, value in model.state_dict().items() if 'running_' in name}
    with torch.no_grad():
        model(*batch())
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())


def test_pieces_with_the_state_carried_are_one_whole_pass():
    from nnslr_tools.brain_targets import drive_chunks
    model = net()
    images, extra = batch()
    with torch.no_grad():
        whole, _works, _state = model(images, extra)
        state, pieces = None, []
        for first, last in drive_chunks(T, 2, 1):
            part, _works, state = model(images[:, first:last], extra[:, first:last], state)
            pieces.append(part)
    assert torch.allclose(torch.cat(pieces, dim=1), whole, atol=1e-5)


def test_the_speed_does_not_depend_on_the_works_head():
    model = net()
    images, extra = batch()
    with torch.no_grad():
        speed, _works, _state = model(images, extra)
        model.works_head.weight.mul_(100).add_(1)
        again, _works, _state = model(images, extra)
    assert torch.equal(speed, again)


def test_the_car_gets_everything_but_the_works_head():
    from nnslr_tools.brain_net import build_brain_net, car_weights
    model = net()
    weights = car_weights(model)
    assert not [name for name in weights if name.startswith('works_head')]
    without = build_brain_net(GROUPS, pretrained=False, works=False)
    assert set(weights) == set(without.state_dict())
    without.load_state_dict(weights)


def test_the_comparison_without_the_works_task_has_no_works_output():
    model = net(works=False)
    _speed, works, _state = model(*batch())
    assert works is None
    backward(model, *batch(), answers(speed_on=[1]))
    assert gradient_of(model, 'gru') > 0



def road_answers(t=T, kinds=2, on=(), inside=()):
    """(answers, weight) (1, t, kinds): on, the (frame, kind) with an answer; inside, those of them inside a mark."""
    answers, weight = torch.zeros(1, t, kinds), torch.zeros(1, t, kinds)
    for frame, kind in on:
        weight[0, frame, kind] = 1
    for frame, kind in inside:
        answers[0, frame, kind] = 1
    return answers, weight


def test_the_kinds_of_road_are_a_logit_a_kind_only_when_asked():
    model = net(road_kinds=3)
    said = model(*batch(), road=True)
    assert len(said) == 4 and said[3].shape == (1, T, 3)
    assert len(model(*batch())) == 3
    assert net()(*batch(), road=True)[3] is None


def test_a_kind_of_road_reaches_the_memory_and_not_the_speed_head():
    from nnslr_tools.brain_net import brain_loss
    model = net(road_kinds=2)
    model.zero_grad()
    speed_logits, works_logit, _state, road_logits = model(*batch(), road=True)
    speed, speed_weight, works, works_weight = answers()
    loss = brain_loss(speed_logits, works_logit, speed, speed_weight, works, works_weight,
                      road=(road_logits, *road_answers(on=[(1, 0), (2, 0)], inside=[(1, 0)])))
    loss.backward()
    assert gradient_of(model, 'road_head') > 0 and gradient_of(model, 'gru') > 0 and gradient_of(model, 'project') > 0
    assert gradient_of(model, 'speed_head') == 0 and gradient_of(model, 'works_head') == 0
    # Only the kind with an answer learns.
    assert float(model.road_head.weight.grad[1].abs().sum()) == 0


def test_each_kind_of_road_weighs_as_the_roadworks():
    from nnslr_tools.brain_net import ROAD_LOSS, WORKS_LOSS, brain_loss
    speed_logits, works_logit, _state, road_logits = net(road_kinds=2)(*batch(), road=True)
    speed, speed_weight, works, works_weight = answers(speed_on=[0])
    alone = brain_loss(speed_logits, works_logit, speed, speed_weight, works, works_weight)
    one = brain_loss(speed_logits, works_logit, speed, speed_weight, works, works_weight,
                     road=(road_logits, *road_answers(on=[(1, 0)])))
    both = brain_loss(speed_logits, works_logit, speed, speed_weight, works, works_weight,
                      road=(road_logits, *road_answers(on=[(1, 0), (1, 1)])))
    nothing = brain_loss(speed_logits, works_logit, speed, speed_weight, works, works_weight,
                         road=(road_logits, *road_answers()))
    assert float(nothing) == pytest.approx(float(alone)) and float(alone) < float(one) < float(both)
    assert ROAD_LOSS == WORKS_LOSS


def test_the_speed_does_not_depend_on_the_road_head():
    model = net(road_kinds=2)
    images, extra = batch()
    with torch.no_grad():
        speed = model(images, extra)[0]
        model.road_head.weight.mul_(100).add_(1)
        again = model(images, extra, road=True)[0]
    assert torch.equal(speed, again)


def test_the_car_gets_neither_the_works_head_nor_the_road_head():
    from nnslr_tools.brain_net import build_brain_net, car_weights
    weights = car_weights(net(road_kinds=4))
    assert not [name for name in weights if name.startswith(('works_head', 'road_head'))]
    without = build_brain_net(GROUPS, pretrained=False, works=False)
    assert set(weights) == set(without.state_dict())
    without.load_state_dict(weights)

def test_the_road_type_is_a_logit_a_type_last_only_when_asked():
    from nnslr_tools.brain_net import outputs
    model = net(road_kinds=2, road_types=4)
    said = model(*batch(), road=True, types=True)
    assert len(said) == 5 and said[3].shape == (1, T, 2) and said[4].shape == (1, T, 4)
    found = outputs(said, road=True, types=True)
    assert found['road'] is said[3] and found['types'] is said[4] and found['speed'] is said[0]
    assert outputs(model(*batch(), types=True), types=True)['types'].shape == (1, T, 4)
    assert len(model(*batch())) == 3 and net()(*batch(), types=True)[3] is None


def type_answers(t=T, on=()):
    """(class, weight) (1, t): on, the (frame, type) with an answer; -1 and no weight elsewhere."""
    found, weight = torch.full((1, t), -1, dtype=torch.long), torch.zeros(1, t)
    for frame, kind in on:
        found[0, frame], weight[0, frame] = kind, 1
    return found, weight


def test_the_road_type_reaches_the_memory_and_not_the_speed_head_and_a_frame_without_one_adds_nothing():
    from nnslr_tools.brain_net import TYPE_LOSS, brain_loss
    model = net(road_types=4)
    model.zero_grad()
    speed_logits, works_logit, _state, type_logits = model(*batch(), types=True)
    speed, speed_weight, works, works_weight = answers()
    alone = brain_loss(speed_logits, works_logit, speed, speed_weight, works, works_weight)
    nothing = brain_loss(speed_logits, works_logit, speed, speed_weight, works, works_weight,
                         types=(type_logits, *type_answers()))
    assert float(nothing) == pytest.approx(float(alone)) and 0 < TYPE_LOSS < 1
    loss = brain_loss(speed_logits, works_logit, speed, speed_weight, works, works_weight,
                      types=(type_logits, *type_answers(on=[(1, 2), (3, 0)])))
    assert float(loss) > float(alone)
    loss.backward()
    assert gradient_of(model, 'type_head') > 0 and gradient_of(model, 'gru') > 0 and gradient_of(model, 'project') > 0
    assert gradient_of(model, 'speed_head') == 0 and gradient_of(model, 'works_head') == 0


def test_the_speed_does_not_depend_on_the_type_head_and_the_car_does_not_get_it():
    from nnslr_tools.brain_net import TRAINING_ONLY, build_brain_net, car_weights
    model = net(road_kinds=2, road_types=4)
    images, extra = batch()
    with torch.no_grad():
        speed = model(images, extra)[0]
        model.type_head.weight.mul_(100).add_(1)
        again = model(images, extra, road=True, types=True)[0]
    assert torch.equal(speed, again) and 'type_head.' in TRAINING_ONLY
    weights = car_weights(model)
    assert not [name for name in weights if name.startswith(('works_head', 'road_head', 'type_head'))]
    without = build_brain_net(GROUPS, pretrained=False, works=False)
    assert set(weights) == set(without.state_dict())


def test_a_network_with_another_input_is_the_same_network():
    from nnslr_tools.brain_net import build_brain_net
    from nnslr_tools.brain_targets import INPUT_SIZES
    groups = tuple(INPUT_SIZES)
    torch.manual_seed(0)
    model = build_brain_net(groups, pretrained=False)
    speed, _works, _state = model(torch.rand(1, 2, 3, 144, 256), torch.rand(1, 2, sum(INPUT_SIZES.values())))
    assert speed.shape[1] == 2


def number_row(kph, known=1.0, ended=0.0, groups=GROUPS):
    from nnslr_tools.brain_targets import BRAIN_INPUTS, INPUT_SIZES
    at = sum(INPUT_SIZES[name] for name in groups[:groups.index('signs')])
    extra = torch.zeros(1, 2, sum(INPUT_SIZES[name] for name in groups))
    extra[..., at], extra[..., at + 1], extra[..., at + 6] = kph / BRAIN_INPUTS.SIGN_KPH_SCALE, known, ended
    return extra


def trusting(model, trust, head=None):
    """A network whose only say is a trust of this much and, with head, one class of its own."""
    with torch.no_grad():
        for layer in (model.speed_head, model.trust):
            layer.weight.zero_()
            layer.bias.zero_()
        model.trust.bias.fill_(trust)
        if head is not None:
            model.speed_head.bias[head] = 1
    return model


@pytest.mark.parametrize('kph', [30, 50, 60, 100, 120, 130])
def test_the_number_in_force_goes_to_its_class_alone_as_much_as_the_network_trusts_it(kph):
    from nnslr_tools.brain_targets import speed_class
    speed = trusting(net(), 3.0)(batch(2)[0], number_row(kph))[0]
    assert speed.argmax(dim=-1).tolist() == [[speed_class(kph)] * 2]
    assert speed.sum(dim=-1).tolist() == [[3.0, 3.0]]


@pytest.mark.parametrize('changes', [{'kph': 0}, {'kph': 100, 'known': 0.0}, {'kph': 100, 'ended': 1.0}])
def test_a_number_not_in_force_goes_nowhere(changes):
    assert trusting(net(), 3.0)(batch(2)[0], number_row(**changes))[0].abs().sum() == 0


def test_the_network_can_say_another_speed_than_the_number():
    from nnslr_tools.brain_targets import speed_class
    model = trusting(net(), 0.5, head=speed_class(120))
    assert model(batch(2)[0], number_row(60))[0].argmax(dim=-1).tolist() == [[speed_class(120)] * 2]
    assert trusting(net(), -1.0)(batch(2)[0], number_row(60))[0][0, 0, speed_class(60)] == -1


def test_the_trust_in_the_number_learns_from_a_frame_that_says_it():
    model = net()
    images, extra = batch(2)[0], number_row(60)
    speed, works, _state = model(images, extra)
    from nnslr_tools.brain_net import brain_loss
    from nnslr_tools.brain_targets import speed_class
    brain_loss(speed, works, torch.full((1, 2), speed_class(60)), torch.ones(1, 2), torch.zeros(1, 2), torch.zeros(1, 2)).backward()
    assert gradient_of(model, 'trust') > 0


def test_the_trust_in_the_number_is_one_value_that_comes_from_the_memory():
    from nnslr_tools.brain_targets import speed_class
    model = net()
    with torch.no_grad():
        model.speed_head.weight.zero_()
        model.speed_head.bias.zero_()
        said = model(batch(2)[0], number_row(60))[0][0, :, speed_class(60)]
    assert said[0] != said[1]
    assert model.trust.out_features == 1


def test_a_mark_does_not_reach_the_trust_through_the_number():
    model = net()
    backward(model, batch(2)[0], number_row(60), answers(2, works_on=[0, 1]))
    assert gradient_of(model, 'works_head') > 0
    assert gradient_of(model, 'trust') == 0


def test_with_no_signs_among_the_groups_there_is_no_number_to_add():
    from nnslr_tools.brain_net import build_brain_net
    from nnslr_tools.brain_targets import INPUT_SIZES
    groups = ('plate', 'hour', 'country')
    model = trusting(build_brain_net(groups, pretrained=False), 3.0)
    extra = torch.ones(1, 2, sum(INPUT_SIZES[name] for name in groups))
    assert model(batch(2)[0], extra)[0].abs().sum() == 0


def test_the_number_is_found_wherever_the_signs_are_among_the_groups():
    from nnslr_tools.brain_net import build_brain_net
    from nnslr_tools.brain_targets import speed_class
    groups = ('plate', 'hour', 'signs', 'country')
    model = trusting(build_brain_net(groups, pretrained=False), 3.0)
    assert model(batch(2)[0], number_row(80, groups=groups))[0].argmax(dim=-1).tolist() == [[speed_class(80)] * 2]


def test_the_car_gets_the_trust():
    from nnslr_tools.brain_net import car_weights
    assert 'trust.weight' in car_weights(net())


def test_a_crossing_weighs_with_its_pos_weight_and_every_other_kind_as_before_to_the_last_bit():
    from nnslr_tools.brain_net import ROAD_EVENT_POS_WEIGHT_MAX, brain_loss
    speed_logits, works_logit, _state, road_logits = net(road_kinds=3)(*batch(), road=True)
    labels = answers(speed_on=[0])

    def loss(road):
        return brain_loss(speed_logits, works_logit, *labels, frames=T, road=road)

    # The kinds of before answered, the crossing (column 2) not: no pos_weight, ones, or the crossing's alone.
    old = road_answers(kinds=3, on=[(0, 0), (1, 0), (2, 1), (3, 1)], inside=[(1, 0), (2, 1)])
    plain = loss((road_logits, *old))
    assert torch.equal(plain, loss((road_logits, *old, None)))
    assert torch.equal(plain, loss((road_logits, *old, torch.ones(3))))
    assert torch.equal(plain, loss((road_logits, *old, torch.tensor([1.0, 1.0, 40.0]))))
    # A frame of a crossing: lifted, it weighs more; a frame answered "no crossing" weighs as it did.
    yes = road_answers(kinds=3, on=[(0, 0), (1, 0), (2, 1), (3, 1), (4, 2)], inside=[(1, 0), (2, 1), (4, 2)])
    lift = torch.tensor([1.0, 1.0, 40.0])
    assert float(loss((road_logits, *yes, lift))) > float(loss((road_logits, *yes)))
    no = road_answers(kinds=3, on=[(0, 0), (1, 0), (2, 1), (3, 1), (4, 2)], inside=[(1, 0), (2, 1)])
    assert float(loss((road_logits, *no, lift))) == pytest.approx(float(loss((road_logits, *no))))
    # Of the yes, pos_weight times the error of the plain loss on that frame.
    from nnslr_tools.brain_net import ROAD_LOSS
    alone = road_answers(kinds=3, on=[(4, 2)], inside=[(4, 2)])
    base = loss((road_logits, torch.zeros(1, T, 3), torch.zeros(1, T, 3)))
    one = float(loss((road_logits, *alone)) - base)
    assert float(loss((road_logits, *alone, lift)) - base) == pytest.approx(40 * one, rel=1e-4) and one > 0
    assert ROAD_LOSS == 0.3 and ROAD_EVENT_POS_WEIGHT_MAX == 100.0


def test_a_lifted_crossing_reaches_the_memory_and_only_its_own_row_of_the_road_head():
    from nnslr_tools.brain_net import brain_loss
    model = net(road_kinds=3)
    model.zero_grad()
    speed_logits, works_logit, _state, road_logits = model(*batch(), road=True)
    crossing = road_answers(kinds=3, on=[(2, 2)], inside=[(2, 2)])
    loss = brain_loss(speed_logits, works_logit, *answers(), road=(road_logits, *crossing, torch.tensor([1.0, 1.0, 9.0])))
    loss.backward()
    assert gradient_of(model, 'road_head') > 0 and gradient_of(model, 'gru') > 0
    assert gradient_of(model, 'speed_head') == 0
    assert float(model.road_head.weight.grad[:2].abs().sum()) == 0 and float(model.road_head.weight.grad[2].abs().sum()) > 0


def test_no_network_has_the_road_users_head_any_more_and_its_name_stays_out_of_the_car():
    # users_head learnt the road users from 12 to 17 Nov 2026 (#112 took it away): its brains load without it
    # (test_brain_export), and TRAINING_ONLY keeps its name for them.
    from nnslr_tools import brain_net
    from nnslr_tools.brain_net import TRAINING_ONLY, build_brain_net, outputs
    with pytest.raises(TypeError):
        build_brain_net(GROUPS, pretrained=False, users=True)
    model = net(road_kinds=2, road_types=4)
    assert not hasattr(model, 'users_head') or model.users_head is None
    assert 'users_head.' in TRAINING_ONLY and not hasattr(brain_net, 'USERS_LOSS')
    said = model(*batch(), road=True, types=True)
    assert len(said) == 5 and set(outputs(said, road=True, types=True)) == {'speed', 'works', 'state', 'road', 'types'}


MAP_GROUPS = (*GROUPS, 'map_limit')


def map_row(kph, sign=None):
    """(1, 2, inputs) of a brain with the map: the map's limit (None: no map) and, with sign, that number in force."""
    from nnslr_tools.brain_targets import BRAIN_INPUTS, INPUT_SIZES
    extra = number_row(sign or 0, known=1.0 if sign else 0.0, groups=MAP_GROUPS)
    at = sum(INPUT_SIZES[name] for name in MAP_GROUPS[:-1])
    extra[..., at:at + 2] = torch.from_numpy(BRAIN_INPUTS.map_limit_row(kph))
    return extra


def map_net(trust_map=None):
    from nnslr_tools.brain_net import build_brain_net
    torch.manual_seed(0)
    model = build_brain_net(MAP_GROUPS, pretrained=False, map_path=True).eval()
    if trust_map is not None:
        with torch.no_grad():
            model.trust_map.bias.fill_(trust_map)
    return model


def test_the_maps_path_starts_at_zero_so_a_brain_starts_saying_what_it_said_without_it():
    """The first brain with the map hardly read it as one value among the inputs (14 Nov 2026): it has trust_map since,
    which adds nothing until it has learnt to."""
    model = map_net()
    assert not model.trust_map.weight.any() and not model.trust_map.bias.any()
    out = torch.rand(1, 2, 128)
    with torch.no_grad():
        assert torch.equal(model.speed(out, map_row(80)), model.speed(out, map_row(None)))
    assert model(batch(2)[0], map_row(80))[0].shape == (1, 2, 200)


@pytest.mark.parametrize('kph', [30, 50, 80, 120])
def test_the_maps_trust_goes_to_the_class_of_the_maps_limit_alone(kph):
    from nnslr_tools.brain_targets import speed_class
    model, images = map_net(trust_map=3.0), batch(2)[0]
    with torch.no_grad():
        added = model(images, map_row(kph))[0] - model(images, map_row(None))[0]
    assert added[0, :, speed_class(kph)].tolist() == pytest.approx([3.0, 3.0], abs=0.05)
    added[0, :, speed_class(kph)] = 0
    # The map is also one of the memory's inputs, so the other classes may move a little; the trust is on one class.
    assert added.abs().max() < 1.0
    assert model.map_number(map_row(kph)).sum(dim=-1).tolist() == [[1.0, 1.0]]
    assert not model.map_number(map_row(None)).any()


def test_the_sign_and_the_map_each_add_their_trust():
    from nnslr_tools.brain_targets import speed_class
    model = trusting(map_net(trust_map=2.0), 3.0)
    with torch.no_grad():
        both = model(batch(2)[0], map_row(80, sign=50))[0][0, 0]
        same = model(batch(2)[0], map_row(50, sign=50))[0][0, 0]
    assert both[speed_class(50)] == pytest.approx(3.0) and both[speed_class(80)] == pytest.approx(2.0)
    assert same[speed_class(50)] == pytest.approx(5.0)


def test_the_maps_path_needs_the_map_among_the_groups_and_goes_to_the_car():
    from nnslr_tools.brain_net import build_brain_net, car_weights
    with pytest.raises(ValueError, match='map_limit'):
        build_brain_net(GROUPS, pretrained=False, map_path=True)
    assert 'trust_map.weight' in car_weights(map_net())
    assert build_brain_net(MAP_GROUPS, pretrained=False).trust_map is None  # the brain of 14 Nov 20:23: the input alone


# -- the number path (#104, 16 Nov 2026): the number in force on its class alone, or as a bell around it ----------------
def bell_at(model, kph):
    """A network whose speed_head says a bell around kph whatever the memory holds, and whose trust is its bias alone."""
    from nnslr_tools.brain_net import SPEED_SIGMA_KPH
    trusting(model, 0.0)
    with torch.no_grad():
        model.speed_head.bias.copy_(-0.5 * ((model.class_kph - kph) / SPEED_SIGMA_KPH) ** 2)
    return model.eval()


def best_trust(model, his):
    """(the loss with no trust, the least loss any trust reaches, that trust, the speed said with it) on a frame with the
    number 60 in force and his speed as the answer: trust from -5 to 40 in steps of 0.05."""
    from nnslr_tools.brain_net import brain_loss
    from nnslr_tools.brain_targets import speed_class
    out, extra = torch.zeros(1, 1, 128), number_row(60)[:, :1]
    answer = (torch.tensor([[speed_class(his)]]), torch.ones(1, 1), torch.zeros(1, 1), torch.zeros(1, 1))
    found = []
    with torch.no_grad():
        for trust in torch.linspace(-5, 40, 901):
            model.trust.bias.fill_(float(trust))
            logits = model.speed(out, extra)
            found.append((float(brain_loss(logits, None, *answer)), float(trust), float(model.class_kph[int(logits.argmax())])))
        model.trust.bias.fill_(0.0)
        none = float(brain_loss(model.speed(out, extra), None, *answer))
    return (none, *min(found))


def test_the_bell_path_takes_the_loss_where_the_class_path_cannot_and_moves_the_speed_towards_the_number():
    """#104: speed_head says 50 (the scene), the sign says 60 and he drives 60. The answer is a bell 2 km/h wide on a
    class a km/h, of which one class holds 0.2: a trust added to the class of the 60 alone takes the loss from 14.61 to
    12.30 and no further, so little gradient reaches it. The same trust on a bell around the 60 reaches 5.78 with
    speed_head as it is, and the speed said moves from 50 to 58."""
    none, least, trust, said = best_trust(bell_at(net(), 50), 60)
    assert (none, least) == (pytest.approx(14.61, abs=0.01), pytest.approx(12.30, abs=0.01))
    assert said == 60 and trust == pytest.approx(12.7, abs=0.1)  # one class lifted above the bell at 50, nothing between
    none, least_bell, trust, said = best_trust(bell_at(net(number_path='bell'), 50), 60)
    assert (none, least_bell) == (pytest.approx(14.61, abs=0.01), pytest.approx(5.78, abs=0.01))
    assert least_bell < least - 6
    assert said == 58 and 50 < said <= 60 and trust == pytest.approx(4.5, abs=0.1)


def test_a_trust_of_one_on_the_bell_path_is_the_answer_of_a_driver_who_holds_the_number():
    """With speed_head saying nothing, the bell at trust 1 scores the least an answer can: the answer's own entropy
    (2.112 at SPEED_SIGMA_KPH 2). The class path cannot come near it at any trust."""
    from nnslr_tools.brain_net import brain_loss, speed_targets
    from nnslr_tools.brain_targets import speed_class
    answer = speed_targets(torch.tensor([speed_class(60)]))
    floor = float(torch.special.entr(answer).sum())
    model = trusting(net(number_path='bell'), 1.0).eval()
    with torch.no_grad():
        said = model.speed(torch.zeros(1, 1, 128), number_row(60)[:, :1])
    loss = brain_loss(said, None, torch.tensor([[speed_class(60)]]), torch.ones(1, 1), torch.zeros(1, 1), torch.zeros(1, 1))
    assert float(loss) == pytest.approx(floor, abs=1e-3) and floor == pytest.approx(2.112, abs=1e-3)
    assert best_trust(trusting(net(), 0.0).eval(), 60)[1] > 4.7  # the class path, speed_head silent: 4.74 at its best


def test_a_frame_that_says_the_number_pushes_the_bells_trust_far_harder_than_the_class_paths():
    """Why the trust of the class path stayed near its random start (#104): from a brain that knows nothing yet, a frame
    where he drives the number in force asks 0.19 of the trust on one class (the share of the answer that class holds,
    less 1 in 200) and 16 of the trust on the bell, 80 times as much."""
    from nnslr_tools.brain_net import brain_loss
    from nnslr_tools.brain_targets import speed_class
    pushes = {}
    for path in ('class', 'bell'):
        model = trusting(net(number_path=path), 0.0)
        said = model.speed(torch.zeros(1, 1, 128), number_row(60)[:, :1])
        brain_loss(said, None, torch.tensor([[speed_class(60)]]), torch.ones(1, 1), torch.zeros(1, 1), torch.zeros(1, 1)).backward()
        pushes[path] = -float(model.trust.bias.grad)  # above 0: the loss falls as the trust rises
    assert pushes['class'] == pytest.approx(0.19, abs=0.01) and pushes['bell'] == pytest.approx(16.1, abs=0.1)
    assert pushes['bell'] > 80 * pushes['class']


@pytest.mark.parametrize('kph', [30, 50, 60, 100, 120, 130])
def test_the_bell_path_adds_the_bell_around_the_number_and_never_less_than_its_floor(kph):
    from nnslr_tools.brain_net import NUMBER_FLOOR, SPEED_SIGMA_KPH
    from nnslr_tools.brain_targets import speed_class
    model = trusting(net(number_path='bell'), 1.0).eval()
    with torch.no_grad():
        speed = model(batch(2)[0], number_row(kph))[0][0, 0]
    wanted = (-(model.class_kph - kph) ** 2 / (2 * SPEED_SIGMA_KPH ** 2)).clamp_min(NUMBER_FLOOR)
    assert torch.allclose(speed, wanted, atol=1e-4)
    assert int(speed.argmax()) == speed_class(kph) and float(speed.max()) == pytest.approx(0.0, abs=1e-5)
    # 12 km/h from the number and beyond, every class gets the floor: speed_head still says which of them.
    assert float(speed.min()) == NUMBER_FLOOR == -18.0
    assert speed[speed_class(kph) - 12].item() == NUMBER_FLOOR and speed[speed_class(kph) - 11].item() > NUMBER_FLOOR
    # As much as the network trusts it, and against it with a trust below zero.
    with torch.no_grad():
        doubted = trusting(net(number_path='bell'), -0.5).eval()(batch(2)[0], number_row(kph))[0][0, 0]
    assert torch.allclose(doubted, -0.5 * wanted, atol=1e-4)


@pytest.mark.parametrize('changes', [{'kph': 0, 'known': 0.0}, {'kph': 100, 'known': 0.0}, {'kph': 100, 'ended': 1.0}])
def test_a_number_not_in_force_adds_no_bell(changes):
    # No number yet (the car's row is all zeros then), a number not known, an end read since.
    assert trusting(net(number_path='bell'), 3.0)(batch(2)[0], number_row(**changes))[0].abs().sum() == 0


@pytest.mark.parametrize('kph', [0.0, 250.0, 1000.0, 60000.0])
def test_no_value_of_the_bell_is_out_of_a_half_floats_reach_whatever_number_comes_in(kph):
    """The comma runs the network with half floats, which end at 65504: the distance is cut at 12 km/h before its
    square, so the bell stays within 0 and the floor for a number no sign says too (the square first, 1000 km/h from a
    class, would be a million)."""
    from nnslr_tools.brain_net import NUMBER_FLOOR
    model = net(number_path='bell').eval()
    with torch.no_grad():
        added = model.number_bell(number_row(kph))
    assert float(added.max()) <= 0.0 and float(added.min()) >= NUMBER_FLOOR
    assert torch.isfinite(added.half()).all()


def test_the_trust_of_the_bell_path_starts_at_zero_and_learns_from_a_frame_that_says_the_number():
    from nnslr_tools.brain_net import brain_loss
    from nnslr_tools.brain_targets import speed_class
    model = net(number_path='bell')
    assert model.number_path == 'bell' and not model.trust.weight.any() and not model.trust.bias.any()
    images, extra = batch(2)[0], number_row(60)
    # At zero the number adds nothing: with a number in force the brain says what its speed_head says.
    out = torch.rand(1, 2, 128)
    with torch.no_grad():
        assert torch.equal(model.speed(out, extra), model.speed_head(out))
    speed, works, _state = model(images, extra)
    brain_loss(speed, works, torch.full((1, 2), speed_class(60)), torch.ones(1, 2), torch.zeros(1, 2), torch.zeros(1, 2)).backward()
    assert gradient_of(model, 'trust') > 0
    # The path of every brain before starts where it always did.
    assert net().number_path == 'class' and net().trust.weight.any()


def test_a_number_path_is_one_of_the_two_and_the_limit_ahead_has_only_the_class():
    from nnslr_tools.brain_net import NUMBER_PATHS, build_brain_net, car_weights, number_path_of
    assert NUMBER_PATHS == ('class', 'bell')
    with pytest.raises(ValueError, match='number_path'):
        build_brain_net(GROUPS, pretrained=False, number_path='gauss')
    with pytest.raises(ValueError, match='limit ahead'):
        build_brain_net(GROUPS, pretrained=False, ahead=True, number_path='bell')
    # The word of a checkpoint: one that does not say is of the class path, one this code does not know is refused.
    assert number_path_of({}) == 'class' and number_path_of({'number_path': None}) == 'class'
    assert number_path_of({'number_path': 'bell'}) == 'bell' and number_path_of({'number_path': 'class'}) == 'class'
    with pytest.raises(ValueError, match='number path'):
        number_path_of({'number_path': 'gauss'})
    # The same weights for the car, by the same names: the path is arithmetic, not a layer.
    assert set(car_weights(net(number_path='bell'))) == set(car_weights(net()))


def test_the_maps_limit_of_a_brain_of_the_bell_path_adds_its_own_bell():
    from nnslr_tools.brain_net import NUMBER_FLOOR, SPEED_SIGMA_KPH, build_brain_net
    from nnslr_tools.brain_targets import speed_class
    torch.manual_seed(0)
    model = build_brain_net(MAP_GROUPS, pretrained=False, map_path=True, number_path='bell').eval()
    assert not model.trust_map.weight.any() and not model.trust_map.bias.any()
    trusting(model, 3.0)
    with torch.no_grad():
        model.trust_map.bias.fill_(2.0)
        out = torch.zeros(1, 2, 128)
        both, same, none = (model.speed(out, row)[0, 0] for row in (map_row(80, sign=50), map_row(50, sign=50), map_row(None)))

    def bell(kph):
        return (-(model.class_kph - kph) ** 2 / (2 * SPEED_SIGMA_KPH ** 2)).clamp_min(NUMBER_FLOOR)

    assert torch.allclose(both, 3.0 * bell(50) + 2.0 * bell(80), atol=1e-4)
    assert torch.allclose(same, 5.0 * bell(50), atol=1e-4) and int(same.argmax()) == speed_class(50)
    assert not none.any() and not model.map_bell(map_row(None)).any()


# -- whose road a sign is of (#89): applicability_head, a logit a column of sign_boxes ------------------------------------
BOX_GROUPS = (*GROUPS, 'sign_boxes')


def box_net(**options):
    from nnslr_tools.brain_net import build_brain_net
    torch.manual_seed(0)
    model = build_brain_net(BOX_GROUPS, pretrained=False, **options)
    model.train()
    for layer in model.modules():
        if isinstance(layer, torch.nn.Dropout):
            layer.p = 0.0
    return model


def box_batch(t=T):
    from nnslr_tools.brain_targets import INPUT_SIZES
    torch.manual_seed(1)
    return torch.rand(1, t, 3, 144, 256), torch.rand(1, t, SIZE + INPUT_SIZES['sign_boxes'])


def whose_answers(ours=(), another=()):
    """(answers, weight) (1, T, 8): 1 of another road, on the (frame, column) cells a reviewed box answers."""
    answer, weight = torch.zeros(1, T, 8), torch.zeros(1, T, 8)
    for frame, column in (*ours, *another):
        weight[0, frame, column] = 1
    for frame, column in another:
        answer[0, frame, column] = 1
    return answer, weight


def test_whose_road_a_sign_is_of_is_a_logit_a_column_of_the_sign_boxes_last_only_when_asked():
    from nnslr_tools.brain_net import APPLICABILITY_COLUMNS, build_brain_net, outputs
    from nnslr_tools.brain_targets import BRAIN_INPUTS
    model = box_net(road_types=4, applicability=True)
    said = model(*box_batch(), types=True, applicability=True)
    # A brain of sign_boxes (until 17 Nov 2026) keeps a logit a column; one of sign_slots has one a slot (#110)
    assert len(said) == 5 and said[4].shape == (1, T, BRAIN_INPUTS.BOX_COLUMNS) and BRAIN_INPUTS.BOX_COLUMNS == 8
    assert APPLICABILITY_COLUMNS == BRAIN_INPUTS.SIGN_SLOTS == 16
    slotted = build_brain_net((*GROUPS, 'sign_slots'), pretrained=False, applicability=True)
    assert slotted.applicability_head.out_features == 16
    found = outputs(said, types=True, applicability=True)
    assert found['applicability'] is said[4] and found['types'] is said[3]
    assert len(model(*box_batch())) == 3 and box_net()(*box_batch(), applicability=True)[3] is None
    # It answers of the columns of sign_boxes: a brain without them cannot learn it
    with pytest.raises(ValueError, match='sign_slots'):
        build_brain_net(GROUPS, pretrained=False, applicability=True)


def test_a_reviewed_sign_reaches_the_memory_and_not_the_speed_head_and_a_column_nobody_reviewed_adds_nothing():
    from nnslr_tools.brain_net import APPLICABILITY_LOSS, brain_loss
    model = box_net(applicability=True)
    model.zero_grad()
    speed_logits, works_logit, _state, whose = model(*box_batch(), applicability=True)
    speed, speed_weight, works, works_weight = answers()
    alone = brain_loss(speed_logits, works_logit, speed, speed_weight, works, works_weight)
    # No reviewed box anywhere: the same loss as without the head, whatever the answers say.
    nothing = brain_loss(speed_logits, works_logit, speed, speed_weight, works, works_weight,
                         applicability=(whose, torch.ones(1, T, 8), torch.zeros(1, T, 8), None))
    assert float(nothing) == pytest.approx(float(alone))
    labels = whose_answers(ours=[(1, 6)], another=[(3, 0)])
    loss = brain_loss(speed_logits, works_logit, speed, speed_weight, works, works_weight, applicability=(whose, *labels, None))
    assert float(loss) > float(alone)
    # With the frames of a piece given, a reviewed sign weighs as a frame of his speed: the sum of its errors over them.
    errors = torch.nn.functional.binary_cross_entropy_with_logits(whose, labels[0], reduction='none')
    by_frames = brain_loss(speed_logits, works_logit, speed, speed_weight, works, works_weight, frames=10,
                           applicability=(whose, *labels, None))
    alone_by_frames = brain_loss(speed_logits, works_logit, speed, speed_weight, works, works_weight, frames=10)
    assert float(by_frames - alone_by_frames) == pytest.approx(APPLICABILITY_LOSS * float((errors * labels[1]).sum()) / 10, rel=1e-5)
    loss.backward()
    assert gradient_of(model, 'applicability_head') > 0 and gradient_of(model, 'gru') > 0 and gradient_of(model, 'project') > 0
    assert gradient_of(model, 'speed_head') == 0 and gradient_of(model, 'works_head') == 0
    # Only the answered columns learn.
    grad = model.applicability_head.weight.grad
    assert float(grad[6].abs().sum()) > 0 and float(grad[0].abs().sum()) > 0 and float(grad[3].abs().sum()) == 0


def test_a_sign_of_another_road_weighs_more_with_its_pos_weight_and_one_of_ours_the_same():
    from nnslr_tools.brain_net import brain_loss
    speed_logits, works_logit, _state, whose = box_net(applicability=True)(*box_batch(), applicability=True)
    speed, speed_weight, works, works_weight = answers()

    def loss(labels, lift):
        return float(brain_loss(speed_logits, works_logit, speed, speed_weight, works, works_weight,
                                applicability=(whose, *labels, lift)))

    another, ours = whose_answers(another=[(3, 0)]), whose_answers(ours=[(1, 6)])
    assert loss(another, torch.tensor(5.0)) > loss(another, None)
    assert loss(ours, torch.tensor(5.0)) == pytest.approx(loss(ours, None))


def test_the_speed_does_not_depend_on_the_applicability_head_and_the_car_does_not_get_it():
    from nnslr_tools.brain_net import TRAINING_ONLY, build_brain_net, car_weights
    model = box_net(applicability=True)
    images, extra = box_batch()
    with torch.no_grad():
        speed = model(images, extra)[0]
        model.applicability_head.weight.mul_(100).add_(1)
        again = model(images, extra, applicability=True)[0]
    assert torch.equal(speed, again) and 'applicability_head.' in TRAINING_ONLY
    without = build_brain_net(BOX_GROUPS, pretrained=False, works=False)
    assert set(car_weights(model)) == set(without.state_dict())


def test_the_bell_of_the_number_path_keeps_its_own_width(monkeypatch):
    """#104: a brain of the 'bell' path is built again from its word alone, so the bell's width and floor are the
    path's own (NUMBER_SIGMA_KPH, NUMBER_FLOOR): another width of the answer's bell (SPEED_SIGMA_KPH, the loss's) must
    not change what a trained brain says."""
    from nnslr_tools import brain_net
    assert brain_net.NUMBER_SIGMA_KPH == 2.0 and brain_net.NUMBER_FLOOR == -18.0

    def bell():
        extra = torch.zeros(1, 1, SIZE)
        extra[..., 0], extra[..., 1] = 60 / brain_net.BRAIN_INPUTS.SIGN_KPH_SCALE, 1.0  # signs is the first group
        return net(number_path='bell').number_bell(extra)[0, 0]

    before = bell()
    assert before.tolist()[59] == 0.0 and before.tolist()[61] == pytest.approx(-0.5)  # the classes of 60 and 62 km/h
    assert before.tolist()[71] == pytest.approx(-18.0) and before.tolist()[119] == pytest.approx(-18.0)
    monkeypatch.setattr(brain_net, 'SPEED_SIGMA_KPH', 3.0)
    assert torch.equal(bell(), before)
