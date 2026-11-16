import pytest

from relu_sat_bench.cnf import CNF, brute_force_max_satisfied, iter_assignments


def test_from_clauses_infers_num_vars():
    cnf = CNF.from_clauses([[1, -3], [2]])
    assert cnf.num_vars == 3
    assert cnf.clauses == ((1, -3), (2,))
    assert cnf.num_clauses == 2


@pytest.mark.parametrize("bad_clauses", [((0,),), ((4,),), ((1, -4),)])
def test_rejects_invalid_literals(bad_clauses):
    with pytest.raises(ValueError):
        CNF(3, bad_clauses)


def test_count_satisfied_total_and_partial():
    cnf = CNF(3, ((1, -2), (2, 3), (-1, -3)))
    assert cnf.count_satisfied([True, True, False]) == 3
    assert cnf.is_satisfied_by([True, True, False])
    assert cnf.count_satisfied([True, False, True]) == 2
    # Unassigned variables make no literal true.
    assert cnf.satisfied_clauses([None, None, None]) == [False, False, False]
    assert cnf.satisfied_clauses([True, None, None]) == [True, False, False]


def test_assignment_length_is_checked():
    with pytest.raises(ValueError):
        CNF(2, ((1, 2),)).count_satisfied([True])


def test_dimacs_round_trip():
    cnf = CNF(4, ((1, -2, 3), (-4,), (2, 4, -1)))
    text = cnf.to_dimacs(comments=["example"])
    assert text.startswith("c example\np cnf 4 3\n")
    assert CNF.from_dimacs(text) == cnf


def test_dimacs_parser_handles_multiline_clauses_and_satlib_terminator():
    text = "c comment\np cnf 3 2\n1 -2\n 3 0 -1\n0\n%\n0\n"
    assert CNF.from_dimacs(text) == CNF(3, ((1, -2, 3), (-1,)))


@pytest.mark.parametrize(
    "text",
    [
        "1 2 0\n",  # no header
        "p cnf 2 1\n1 2\n",  # clause not terminated
        "p cnf 2 2\n1 2 0\n",  # wrong clause count
        "p dnf 2 1\n1 2 0\n",  # wrong format
    ],
)
def test_dimacs_parser_rejects_malformed_input(text):
    with pytest.raises(ValueError):
        CNF.from_dimacs(text)


def test_iter_assignments_enumerates_all():
    assignments = list(iter_assignments(3))
    assert len(assignments) == 8
    assert len(set(assignments)) == 8


def test_brute_force_max_satisfied():
    unsat = CNF(1, ((1,), (-1,)))
    assert brute_force_max_satisfied(unsat)[0] == 1
    sat = CNF(2, ((1, 2), (-1,)))
    value, assignment = brute_force_max_satisfied(sat)
    assert value == 2 and sat.is_satisfied_by(assignment)
