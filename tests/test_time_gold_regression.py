"""Time labels must distinguish clock values from transit identifiers."""
import pytest

from evalab import gold as G
from evalab.items import label_sgbus


@pytest.mark.parametrize("identifier", [
    "Service 646", "bus 646", "Route 646", "bus service 646",
    "service no. 646", "route number 646", "bus #646", "Service 1815",
    "Service **646**", "Route `646`",
])
def test_transit_identifiers_do_not_compete_with_departure(identifier):
    assert G.check_time(f"{identifier} first departs at 1815.", "1815") == "correct"


def test_route_number_alone_cannot_supply_the_gold_time():
    assert G.check_time("Service 646", "0646") == "incorrect"
    assert G.check_time("Route 1815", "1815") == "incorrect"
    assert G.check_time("Service 646 departs at 1815.", "0646") == "incorrect"


def test_excluding_identifier_occurrences_preserves_real_times_with_same_value():
    assert G.check_time("Service 646 departs at 0646.", "0646") == "correct"
    assert G.check_time("Service 646 departs at 646.", "0646") == "correct"
    assert G.check_time("Service 646 departs at 6:46 am.", "0646") == "correct"
    assert G.check_time("Service 646 departs at 0646 or 1815.", "1815") == "ambiguous"
    assert G.check_time("Route 1815 departs at 1815 or 1830.", "1815") == "ambiguous"


@pytest.mark.parametrize("answer,gold", [
    ("On weekdays, the first bus of Singapore bus service 646 leaves Hub Synergy Pt (03222) at 1815.", "1815"),
    ("The first bus of service 861M leaves its starting stop Blk 120A (58539) at 0645.", "0645"),
])
def test_transit_label_path_does_not_exclude_unambiguous_correct_answers(answer, gold):
    item = {"kind": "time", "gold": gold, "question": "At what time does the first bus leave? Give HHMM."}
    assert label_sgbus(item, answer) == "correct"


@pytest.mark.parametrize("token", [
    "Blk 120A", "A0530", "0530A", "05300", "05:300", "005:30",
    "105:30", "05:30:15", "1.0530", "0530.25", "x05:30",
])
def test_time_parser_does_not_take_substrings_of_other_tokens(token):
    assert G.hhmm(token) == []


@pytest.mark.parametrize("token", ["24:01", "24:59", "25:00", "05:60", "13:30 pm", "00:30 am"])
def test_time_parser_rejects_invalid_clock_values(token):
    assert G.hhmm(token) == []


@pytest.mark.parametrize("date", [
    "September 2026", "Sep. 2026", "30 September 2026", "September 30, 2026",
    "September 30th, 2026", "year 2026", "2026-09-30", "2026/09/30",
    "30/09/2026", "09/30/2026", "30.09.2026",
])
def test_date_occurrences_do_not_compete_with_clock_values(date):
    assert G.check_time(f"In {date}, the first bus leaves at 0530.", "0530") == "correct"
    assert G.check_time(date, "2026") == "incorrect"


def test_date_filter_preserves_real_times_with_same_value():
    assert G.check_time("In September 2026, it leaves at 2026 (20:26).", "2026") == "correct"
    assert G.check_time("On 2026-09-30, it leaves at 2026 or 20:30.", "2026") == "ambiguous"
    assert G.check_time("The 0530 or 2026 departure in September 2026.", "0530") == "ambiguous"
    assert G.check_time("0530 or 2026", "0530") == "ambiguous"


@pytest.mark.parametrize("token,want", [
    ("0530", "0530"), ("530", "0530"), ("05:30", "0530"),
    ("5.30am", "0530"), ("5:30 a.m.", "0530"), ("5:30PM", "1730"),
    ("12:05 AM", "0005"), ("12:05 p.m.", "1205"),
    ("00:00", "0000"), ("23:59", "2359"), ("24:00", "2400"),
    ("0530hrs", "0530"), ("0530 hours", "0530"), ("**0530**.", "0530"),
    ("Time:0530", "0530"), ("The bus 6:46 am leaves first.", "0646"),
])
def test_supported_time_formats_and_boundaries(token, want):
    assert G.hhmm(token) == [want]
