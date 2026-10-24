"""The labels of our classifier as proposals of the review page (nnslr preannotate --proposer ours)."""
from nnslr_tools._chain_proposer import proposal_of


def test_classifier_labels_become_proposals():
    assert proposal_of('maximum_speed:60') == ('maximum_speed', 60)
    assert proposal_of('maximum_speed:unknown') == ('maximum_speed', None)
    assert proposal_of('variable_display:100') == ('variable_display', 100)
    assert proposal_of('road_marking:30') == ('road_marking_candidate', 30)
    assert proposal_of('other_sign') == ('other_sign', None)
    assert proposal_of('cancellation') == ('cancellation', None)
    assert proposal_of('not_a_sign') is None
