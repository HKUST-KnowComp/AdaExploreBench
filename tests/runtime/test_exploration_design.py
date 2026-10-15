from copy import deepcopy
from itertools import product

import pytest

from protocol.mechanism import FAMILIES
from runtime.exploration_design import (
    DESIGN_VERSION,
    aggregate_exploration_design,
    exploration_design,
)


def episode(family):
    fields = {f: {'minimum': 10, 'maximum': 110} for f in ['a', 'b', 'c', 'noise']}
    rule = dict(family=family, relevant_fields=['a', 'b', 'c'],
                field_directions=dict(a=1, b=1, c=1), thresholds=dict(a=60, b=60, c=60),
                weights=dict(a=1, b=1, c=1), priority_order=['a', 'b', 'c'], indifference_band=.2)
    return {'agent_task': {'fields': fields}, 'evaluation': {'transition': {
        'pre_rule': rule, 'post_rule': deepcopy(rule)}}}


def points(rows, window=0):
    return [{'normalized_values': dict(zip(['a', 'b', 'c', 'noise'], [*r, .5] if len(r) == 3 else r)),
             'window': window} for r in rows]


def score(family, rows):
    return exploration_design(points(rows), episode(family), condition='static', change_after_batch=2)['score']


@pytest.mark.parametrize('family', FAMILIES)
def test_duplicate_order_utility_and_sparse_invariance(family):
    p = points(list(product([0., .5, 1.], repeat=3)))
    e = episode(family)
    def evaluate(rows):
        return exploration_design(rows, e, condition='static', change_after_batch=2)['score']
    expected = evaluate(p)
    assert 0 < expected <= 1
    assert evaluate(p[::-1] + p[:8] * 5) == pytest.approx(expected)
    assert evaluate([{**r, 'utilities': [-1000, 1000]} for r in p]) == expected
    assert evaluate([]) is None
    assert evaluate(p[:1] * 100) == (None if family == 'additive_aggregation' else evaluate(p[:1]))


def test_additive_separates_joint_changes_and_ignores_distractors():
    family = 'additive_aggregation'
    collinear = [(v, v, v) for v in (0, 1/3, 2/3, 1)]
    assert score(family, collinear) == pytest.approx(0)
    corners = list(product([0., 1.], repeat=3))
    assert score(family, corners) == pytest.approx(1)
    assert score(family, [(*r, r[0]) for r in corners]) == pytest.approx(score(family, corners))
    # Pairwise non-collinearity alone does not establish joint independence.
    assert score(family, [(a, b, (a+b)/2) for a, b in product([0., 1.], repeat=2)]) == pytest.approx(0)


@pytest.mark.parametrize('family,background', [('conjunctive_gating', 1.), ('disjunctive_sufficiency', 0.)])
def test_gate_coverage_rewards_complementary_regions(family, background):
    rows = []
    for j in range(3):
        for value in [.1, .9]:
            row = [background]*3
            row[j] = value
            rows.append(tuple(row))
    assert score(family, rows) > score(family, [(0,0,0), (1,1,1)])
    # Same case is worth more in a set which supplies missing field contrasts.
    assert score(family, rows) > score(family, rows[:2])


def test_lexicographic_requires_neutralization_and_conflict():
    family = 'lexicographic_priority'
    shallow = [(0,0,0), (1,0,0)]
    rich = [(0,0,0), (1,0,0), (.5,0,0), (.5,1,0), (.5,.5,0), (.5,.5,1)]
    assert score(family, rich) > score(family, shallow)
    design = exploration_design(points(rich), episode(family), condition='static', change_after_batch=2)
    assert design['phases']['static']['components']['priority_conflict_region_coverage'] > 0


def test_dynamic_boundary_empty_phase_and_post_rule():
    e = episode('additive_aggregation')
    p = points(list(product([0., 1.], repeat=3)))
    before_only = exploration_design(p, e, condition='dynamic', change_after_batch=2)
    # The empty post-change phase is not assessable, so it is excluded rather
    # than scored zero.
    assert before_only['score'] == pytest.approx(1)
    assert before_only['phases']['pre']['valid'] is True
    assert before_only['phases']['post']['valid'] is False
    both = exploration_design(p + [{**r, 'window': 2} for r in p], e, condition='dynamic', change_after_batch=2)
    assert both['score'] == pytest.approx(1)
    e['evaluation']['transition']['post_rule']['relevant_fields'] = ['noise','b','c']
    e['evaluation']['transition']['post_rule']['weights']['noise'] = 1
    changed = exploration_design(p + [{**r, 'window': 2} for r in p], e, condition='dynamic', change_after_batch=2)
    assert changed['score'] < both['score']


def test_threshold_units_and_negative_directions():
    e = episode('conjunctive_gating')
    p = points([(0,1,1), (1,1,1), (1,0,1), (1,1,0), (.8,1,1), (1,.8,1), (1,1,.8)])
    def evaluate():
        return exploration_design(p,e,condition='static',change_after_batch=2)['score']
    expected = evaluate()
    assert expected > 0
    for f in ['a','b','c']:
        e['agent_task']['fields'][f] = {'minimum': -1, 'maximum': 1}
        e['evaluation']['transition']['pre_rule']['thresholds'][f] = 0
    assert evaluate() == pytest.approx(expected)
    e['evaluation']['transition']['pre_rule']['field_directions']['a'] = -1
    for point in p:
        point['normalized_values']['a'] = 1-point['normalized_values']['a']
    assert evaluate() == pytest.approx(expected)


def _aggregate_row(family, score, *, pre=None, post=None, pre_unique=8, post_unique=8):
    phases = {}
    if pre is not None:
        phases['pre'] = {'score': pre, 'unique_probes': pre_unique, 'valid': pre_unique > 0}
    if post is not None:
        phases['post'] = {'score': post, 'unique_probes': post_unique, 'valid': post_unique > 0}
    return {'probe_metrics': {'geometry': {'design': {
        'version': DESIGN_VERSION, 'rule_family': family, 'score': score, 'phases': phases,
    }}}}


def test_aggregation_is_family_balanced_and_reports_validity():
    rows = [_aggregate_row(FAMILIES[0], 1, pre=1, post=1)] * 10
    rows += [_aggregate_row(FAMILIES[1], 0, pre=0, post=0)]
    result = aggregate_exploration_design(rows)
    assert result['exploration_performance'] == .5
    families = result['exploration_by_rule_family']
    assert families[FAMILIES[0]]['score'] == 1.0
    assert families[FAMILIES[0]]['valid_episodes'] == 10
    assert families[FAMILIES[0]]['episode_count'] == 10
    assert families[FAMILIES[1]]['score'] == 0.0
    assert families[FAMILIES[1]]['valid_episodes'] == 1
    assert families[FAMILIES[2]] is None and families[FAMILIES[3]] is None
    assert aggregate_exploration_design([])['exploration_performance'] is None


def test_aggregation_excludes_unassessable_episodes_and_splits_phases():
    rows = [
        _aggregate_row('additive_aggregation', 1.0, pre=1.0, post=1.0),
        _aggregate_row('additive_aggregation', None, pre=0.9, post=0.0, post_unique=0),
        _aggregate_row('additive_aggregation', 0.5, pre=0.6, post=0.4, pre_unique=6, post_unique=4),
    ]
    family = aggregate_exploration_design(rows)['exploration_by_rule_family']['additive_aggregation']
    assert family['score'] == pytest.approx(0.75)
    assert family['valid_episodes'] == 2
    assert family['episode_count'] == 3
    assert family['pre_score'] == pytest.approx((1.0 + 0.9 + 0.6) / 3)
    assert family['pre_valid_episodes'] == 3
    assert family['post_score'] == pytest.approx(0.7)
    assert family['post_valid_episodes'] == 2
    assert family['mean_unique_probes'] == pytest.approx((16 + 8 + 10) / 3)


def test_additive_phase_requires_more_than_d_unique_points():
    e = episode('additive_aggregation')
    three = exploration_design(points([(0,0,0),(1,0,0),(0,1,0)]), e, condition='static', change_after_batch=2)
    assert three['phases']['static']['valid'] is False
    assert three['score'] is None
    four = exploration_design(points([(0,0,0),(1,0,0),(0,1,0),(0,0,1)]), e, condition='static', change_after_batch=2)
    assert four['phases']['static']['valid'] is True
    assert four['score'] > 0


def test_additive_exact_volume_and_relevant_vector_deduplication():
    family = 'additive_aggregation'
    axes = [(0,.5,.5), (1,.5,.5), (.5,0,.5), (.5,1,.5), (.5,.5,0), (.5,.5,1)]
    assert score(family, axes) == pytest.approx(1/3)
    assert score(family, list(product([.25,.75], repeat=3))) == pytest.approx(.25)
    # Three unique points cannot define a 3D covariance volume: unassessable.
    assert score(family, [(0,0,0), (1,0,0), (0,1,0)]) is None
    assert score(family, axes + [(*axes[0], noise) for noise in [0,.2,.8,1]]) == pytest.approx(1/3)
    # A full-rank design at very small scale remains finite and bounded.
    assert score(family, list(product([.5,.500001], repeat=3))) == pytest.approx(1e-12, rel=1e-6, abs=1e-20)


@pytest.mark.parametrize('family', ['conjunctive_gating','disjunctive_sufficiency'])
def test_weighted_regions_exact_values_and_mirror(family):
    baseline_and_single = [(0,0,0),(1,0,0),(0,1,0),(0,0,1)]
    if family == 'conjunctive_gating':
        baseline_and_single = [tuple(1-v for v in row) for row in baseline_and_single]
    assert score(family, baseline_and_single) == pytest.approx(24/35)
    assert score(family, baseline_and_single[:1]) == pytest.approx(6/35)
    assert score(family, list(product([0,1], repeat=3))) == pytest.approx(1)
    for region in product([0,1], repeat=3):
        fails = 3-sum(region)
        expected = (1/max(1,fails))/(35/6)
        assert score('conjunctive_gating',[region]) == pytest.approx(expected)
        assert score('disjunctive_sufficiency',[tuple(1-v for v in region)]) == pytest.approx(expected)
    assert score('conjunctive_gating',[(.5,.5,.5)]) == pytest.approx(2/35)
    assert score('disjunctive_sufficiency',[(.5,.5,.5)]) == pytest.approx(6/35)


def test_lexicographic_exact_signed_regions_and_boundaries():
    family = 'lexicographic_priority'
    regions = [(1,0,.5),(0,1,.5),(.5,1,0),(.5,0,1),(.5,.5,1),(.5,.5,0)]
    for row in regions:
        assert score(family,[row]) == pytest.approx(1/6)
    assert score(family,regions) == pytest.approx(1)
    assert score(family,[(1,1,1),(0,0,0),(.5,.5,.5)]) == 0
    # A neutral lower cue is not conflict; higher decisive cues mask lower ones.
    assert score(family,[(1,.5,.5),(1,1,0)]) == pytest.approx(1/6)
    e = episode(family)
    e['evaluation']['transition']['pre_rule']['indifference_band'] = .25
    block = exploration_design(points([(.625,1,0)]),e,condition='static',change_after_batch=2)
    assert block['phases']['static']['visited_regions'] == ['D_2,+']
    e['evaluation']['transition']['pre_rule']['priority_order'] = ['c','b','a']
    block = exploration_design(points([(1,0,.5)]),e,condition='static',change_after_batch=2)
    assert block['phases']['static']['visited_regions'] == ['D_2,-']


def test_old_design_version_is_excluded():
    row = {'probe_metrics': {'geometry': {'design': {'version': DESIGN_VERSION - 1,'rule_family':FAMILIES[0],'score':1}}}}
    result = aggregate_exploration_design([row])
    assert result['exploration_performance'] is None
    assert result['exploration_by_rule_family'][FAMILIES[0]] is None
