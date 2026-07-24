"""Unit tests for the ingestion URL layout (no network access)."""

from src.ingestion import statsbomb_open_data as sbod


def test_competitions_url_default_ref():
    assert sbod.competitions_url() == (
        "https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json"
    )


def test_matches_url_includes_ids():
    url = sbod.matches_url(competition_id=2, season_id=27)
    assert url.endswith("/data/matches/2/27.json")


def test_lineups_and_events_urls_use_match_id():
    assert sbod.lineups_url(3754117).endswith("/data/lineups/3754117.json")
    assert sbod.events_url(3754117).endswith("/data/events/3754117.json")


def test_ref_is_pinnable():
    url = sbod.events_url(3754117, ref="abc123")
    assert "/open-data/abc123/data/" in url
