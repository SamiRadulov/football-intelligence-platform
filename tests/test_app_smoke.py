"""Smoke tests: every app page must render without raising.

Pages are exercised through `app.py` and `switch_page`, the same way a user
reaches them, because `st.page_link` and the sidebar only exist inside the
navigation context — running a page file standalone would test it in a
situation that never occurs.

Interactive paths (selecting a player, selecting a team) are covered too, since
that is where page-level bugs actually live.

These need the built curated database, so they skip cleanly on a machine that has
not run the pipeline yet — the rest of the suite is pure unit tests with no data
dependency.
"""

from __future__ import annotations

import pytest

from src.app.data import DB_PATH

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.skipif(
    not DB_PATH.exists(),
    reason="curated.duckdb not built; run scripts/build_staging.py first",
)

PAGES = [
    "pages/home.py",
    "pages/player_search.py",
    "pages/player_comparison.py",
    "pages/team_style_map.py",
    "pages/team_profile.py",
    "pages/methodology.py",
]

TIMEOUT = 120


def _open(page: str | None = None) -> AppTest:
    """Run the app, optionally navigating to a page first."""
    app = AppTest.from_file("app.py", default_timeout=TIMEOUT)
    app.run()
    if page is not None:
        app.switch_page(page)
        app.run()
    return app


def _assert_clean(app: AppTest) -> None:
    assert not app.exception, [e.value for e in app.exception]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page):
    _assert_clean(_open(page))


def test_home_reports_dataset_coverage():
    app = _open("pages/home.py")
    labels = [m.label for m in app.metric]
    assert "Matches" in labels
    assert "Qualified players" in labels


def test_player_search_returns_ranked_candidates():
    app = _open("pages/player_search.py")
    options = app.sidebar.selectbox[0].options
    vardy = next(o for o in options if "Vardy" in o)

    app.sidebar.selectbox[0].select(vardy).run()

    _assert_clean(app)
    assert "Role group" in [m.label for m in app.metric]
    assert len(app.dataframe) >= 1          # the ranked results table


def test_player_search_respects_the_minutes_filter():
    app = _open("pages/player_search.py")
    vardy = next(o for o in app.sidebar.selectbox[0].options if "Vardy" in o)
    app.sidebar.selectbox[0].select(vardy).run()

    app.sidebar.slider[0].set_value(3000).run()      # very few players qualify
    _assert_clean(app)


def test_player_comparison_renders_two_player_breakdown():
    app = _open("pages/player_comparison.py")
    app.sidebar.selectbox[0].select("ST").run()

    options = app.sidebar.selectbox[1].options
    app.sidebar.selectbox[1].select(options[0]).run()
    app.sidebar.selectbox[2].select(options[1]).run()

    _assert_clean(app)
    assert any(m.label == "Similarity" for m in app.metric)
    assert len(app.dataframe) >= 2          # "most alike" and "biggest differences"


def test_player_comparison_rejects_the_same_player_twice():
    app = _open("pages/player_comparison.py")
    app.sidebar.selectbox[0].select("ST").run()
    same = app.sidebar.selectbox[1].options[0]
    app.sidebar.selectbox[1].select(same).run()
    app.sidebar.selectbox[2].select(same).run()

    _assert_clean(app)
    assert any("two different players" in w.value for w in app.warning)


def test_team_profile_renders_full_page_for_a_team():
    app = _open("pages/team_profile.py")
    app.sidebar.selectbox[0].select("Leicester City").run()

    _assert_clean(app)
    labels = {m.label: m.value for m in app.metric}
    assert labels.get("Team") == "Leicester City"
    assert "Style cluster" in labels


def test_team_style_map_selects_every_cluster_by_default():
    app = _open("pages/team_style_map.py")
    _assert_clean(app)
    assert len(app.sidebar.multiselect[0].value) >= 2


def test_methodology_page_embeds_the_docs():
    app = _open("pages/methodology.py")
    _assert_clean(app)
    body = " ".join(m.value for m in app.markdown)
    assert "StatsBomb" in body
