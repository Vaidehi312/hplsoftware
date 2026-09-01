"""Two Knowledge Banks in one server process.

A cohort can now be registered and loaded into hpl_kb_test instead of hpl_kb,
and everything the UI reads afterwards has to follow that choice. The failure
this guards against is not an exception — it is a viewer that renders, an
overlay that has numbers, and a chatbot that answers, all from the wrong
database.

Two properties matter more than the plumbing:

  * production is the default everywhere, so a client that has never heard of
    kb_target cannot land test data in the real KB by omission;
  * every process-global cache is keyed by target, because a slide_id is only
    unique *within* a Knowledge Bank.
"""

import re
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

APP = BACKEND.parent / "app" / "app_v28.py"
CLIENT = BACKEND.parent / "app" / "api_client.py"
SERVER_SOURCE = (BACKEND / "tile_server_v2_.py").read_text()


# --- production is the default -------------------------------------------

def test_the_default_target_is_production(_tmp=None):
    """The whole safety argument rests on this. If test were the default, or if
    an unset value fell through to anything else, forgetting the parameter
    would be the dangerous case rather than the harmless one."""
    import tile_server_v2_ as srv

    assert srv._resolve_kb_target(None) == srv.KB_PRODUCTION
    assert srv._resolve_kb_target("") == srv.KB_PRODUCTION
    assert srv.KB_TARGETS[srv.KB_PRODUCTION] == srv.DB_NAME


def test_every_request_model_defaults_to_production(_tmp=None):
    """A client that predates kb_target sends no such field. Pydantic fills the
    default, so that default is what decides where an old UI writes."""
    import tile_server_v2_ as srv

    for model in (srv.RegistrationRequest, srv.KbLoadRequest,
                  srv.KbLoadPreviewRequest, srv.QueryRequest):
        field = model.model_fields.get("kb_target")
        assert field is not None, f"{model.__name__} has no kb_target"
        assert field.default == srv.KB_PRODUCTION, (
            f"{model.__name__}.kb_target defaults to {field.default!r}")


def test_an_unknown_target_is_refused_not_coerced(_tmp=None):
    """Refused rather than silently treated as production: 'prod', 'testing' or
    a typo naming a real database on the same host must not open a connection
    nobody asked for."""
    import tile_server_v2_ as srv
    from fastapi import HTTPException

    for bad in ("prod", "hpl_kb", "testing", "PRODUCTION_", "../other"):
        try:
            srv._resolve_kb_target(bad)
        except HTTPException as e:
            assert e.status_code == 400
            assert "kb_target" in str(e.detail)
        else:
            raise AssertionError(f"{bad!r} was accepted as a target")


def test_target_names_are_case_and_space_insensitive(_tmp=None):
    import tile_server_v2_ as srv
    assert srv._resolve_kb_target("  TEST  ") == srv.KB_TEST
    assert srv._resolve_kb_target("Production") == srv.KB_PRODUCTION


# --- the caches are keyed by target ---------------------------------------

def test_slide_handles_are_keyed_by_target(_tmp=None):
    """A slide_id is unique within a Knowledge Bank, not across them. Keyed on
    slide_id alone, a test lookup would be served production's open file — and
    would render, which is the quietest way to be wrong."""
    assert re.search(r"_wsi_handles:\s*dict\[tuple\[str, str\]",
                     SERVER_SOURCE), "_wsi_handles is not keyed by (target, slide_id)"
    assert re.search(r"_dz_handles:\s*dict\[tuple\[str, str\]",
                     SERVER_SOURCE), "_dz_handles is not keyed by (target, slide_id)"


def test_the_wsi_map_and_heatmap_are_per_target(_tmp=None):
    import tile_server_v2_ as srv
    assert isinstance(srv._wsi_maps, dict)
    assert isinstance(srv._heatmap_probs, dict)
    # and there is no surviving single-database global to fall back to
    assert not hasattr(srv, "_wsi_map"), "the old process-wide _wsi_map is back"


def test_the_engine_registry_builds_one_engine_per_target(_tmp=None):
    """Lazily, and separately. Sharing one engine would send test queries to
    production; building both eagerly would make a missing hpl_kb_test a server
    that will not start."""
    import tile_server_v2_ as srv
    assert isinstance(srv._engines, dict)
    assert "engine" not in dir(srv) or not isinstance(getattr(srv, "engine", None), object.__class__)
    src = SERVER_SOURCE[SERVER_SOURCE.index("def _get_engine"):]
    src = src[:src.index("\ndef ", 1)]
    assert "KB_TARGETS[target]" in src, "the engine URL does not vary by target"


def test_the_image_cache_key_separates_targets_without_breaking_production(_tmp=None):
    """Production must keep its bare slide_id or every JPEG already on disk is
    orphaned; anything else has to be namespaced."""
    import tile_server_v2_ as srv
    assert srv._cache_key("SLIDE-A", srv.KB_PRODUCTION) == "SLIDE-A"
    assert srv._cache_key("SLIDE-A", srv.KB_TEST) != "SLIDE-A"
    assert "SLIDE-A" in srv._cache_key("SLIDE-A", srv.KB_TEST)


# --- the read endpoints honour it -----------------------------------------

_KB_READ_ENDPOINTS = [
    "/health", "/slides", "/slide/{slide_id}/info", "/dzi/{slide_id}.dzi",
    "/slide/{slide_id}/thumbnail", "/slide/{slide_id}/tile",
    "/slide/{slide_id}/region", "/slide/{slide_id}/tiles_meta",
    "/slide/{slide_id}/adjacency", "/hpc/{hpc_id}/info", "/hpc/{hpc_id}/survival",
    "/tile_image/{slide_tile}",
]


def test_every_kb_read_endpoint_accepts_a_target(_tmp=None):
    """Miss one and that panel silently keeps reading production while the rest
    of the screen shows test."""
    import tile_server_v2_ as srv

    routes = {r.path: r for r in srv.app.routes if hasattr(r, "path")}
    missing = []
    for path in _KB_READ_ENDPOINTS:
        route = routes.get(path)
        assert route is not None, f"{path} is not mounted"
        names = {p.name for p in route.dependant.query_params}
        for dep in route.dependant.dependencies:
            names |= {p.name for p in dep.query_params}
        if "kb_target" not in names:
            missing.append(path)
    assert not missing, f"endpoints ignoring kb_target: {missing}"


def test_the_endpoint_check_can_fail(_tmp=None):
    """A route that genuinely has no kb_target must be reported, or the test
    above proves nothing."""
    import tile_server_v2_ as srv

    routes = {r.path: r for r in srv.app.routes if hasattr(r, "path")}
    route = routes.get("/debug/routes")
    assert route is not None
    names = {p.name for p in route.dependant.query_params}
    assert "kb_target" not in names, (
        "/debug/routes now declares kb_target, so this check no longer "
        "distinguishes anything")


# --- the client sends it --------------------------------------------------

def test_the_client_sends_the_target_on_every_get(_tmp=None):
    """Held on the client, not passed per call: app_v28 calls these from far
    more places than there are methods, and threading an argument through each
    is how one call site ends up on the wrong database."""
    source = CLIENT.read_text()
    body = source[source.index("    def _get(self"):]
    body = body[:body.index("\n    def ", 1)]
    assert '"kb_target": self.kb_target' in body, (
        "_get does not attach kb_target, so read endpoints fall back to production")


def test_the_write_methods_send_the_target_in_the_body(_tmp=None):
    """GET params do not reach a POST body. These four are the writes, so
    missing one means a cohort registered into production while the UI said
    test."""
    source = CLIENT.read_text()
    for method in ("preview_registration", "commit_registration",
                   "preview_kb_load", "commit_kb_load"):
        start = source.index(f"def {method}(")
        body = source[start:source.index("\n    def ", start + 1)]
        assert '"kb_target": self.kb_target' in body, (
            f"api_client.{method} does not send kb_target")


def test_commit_registration_sends_the_subset_it_was_given(_tmp=None):
    """It accepted scope and slide_names and then left them out of the body, so
    a subset previewed as three slides committed as the whole dataset —
    silently, because registering more than intended still succeeds."""
    source = CLIENT.read_text()
    start = source.index("def commit_registration(")
    body = source[start:source.index("\n    def ", start + 1)]
    for field in ('"scope": scope', '"slide_names": slide_names'):
        assert field in body, f"commit_registration drops {field}"


def test_switching_target_clears_the_image_cache(_tmp=None):
    source = CLIENT.read_text()
    start = source.index("def set_kb_target(")
    body = source[start:source.index("\n    def ", start + 1)]
    assert "self.cache.clear()" in body, (
        "switching target keeps images cached by slide_id, which means test "
        "can show production's tiles")


# --- the Streamlit app ----------------------------------------------------

def test_the_app_points_its_direct_engine_at_the_selected_target(_tmp=None):
    """The chatbot and three viewer helpers bypass the API entirely. If only the
    API client followed the selector, the pipeline would write to test and every
    answer would still come from production."""
    source = APP.read_text()
    assert "DB_NAME = KB_DATABASES[kb_target]" in source, (
        "app_v28's direct SQLAlchemy engine is not built from the selected target")


def test_the_app_clears_its_caches_when_the_target_changes(_tmp=None):
    """Most @st.cache_data readers here take no arguments, so their key does not
    include the target and they would serve the other database for the ttl."""
    source = APP.read_text()
    assert "st.cache_data.clear()" in source
    assert "_kb_target_active" in source


def test_the_selector_is_resolved_before_the_client_and_engine(_tmp=None):
    """Order matters: both consumers are built at import time, so a selector
    below either of them would leave that one on the previous rerun's choice."""
    source = APP.read_text()
    selector = source.index('key="kb_target"')
    assert selector < source.index("client = TileServerClient(")
    assert selector < source.index("DB_NAME = KB_DATABASES[kb_target]")


# --- path overrides for runs the record cannot describe -------------------


def test_the_override_fields_default_to_none(_tmp=None):
    """None means "take it from the run". Any other default would make every
    existing caller start overriding something."""
    import tile_server_v2_ as srv
    for name in ("dataset_name", "raw_dir", "tile_dir", "h5_path"):
        field = srv.RegistrationRequest.model_fields.get(name)
        assert field is not None, f"RegistrationRequest has no {name}"
        assert field.default is None, f"{name} defaults to {field.default!r}"


def test_the_client_sends_the_overrides_on_both_methods(_tmp=None):
    """Preview accepting an override that commit drops is how a previewed
    directory becomes a different registered one — the same shape as the
    scope/slide_names bug."""
    source = CLIENT.read_text()
    for method in ("preview_registration", "commit_registration"):
        start = source.index(f"def {method}(")
        body = source[start:source.index("\n    def ", start + 1)]
        for field in ("dataset_name", "raw_dir", "tile_dir", "h5_path"):
            assert f'"{field}": {field}' in body, f"{method} drops {field}"


def test_the_app_sends_the_same_overrides_to_preview_and_commit(_tmp=None):
    source = APP.read_text()
    assert source.count("**_overrides,") == 2, (
        "preview and commit must send the same overrides, or the preview "
        "describes a registration that is not the one performed")


def test_the_overrides_dict_is_actually_built(_tmp=None):
    """Counting the uses is not enough: an edit added both `**_overrides` call
    sites while the block that builds it failed to apply, so the app parsed
    cleanly and raised NameError the moment anyone opened step 5. Streamlit
    files are not imported by this suite, so nothing else would have caught it.
    """
    source = APP.read_text()
    assert "_overrides = dict(" in source, "_overrides is used but never assigned"
    assert source.index("_overrides = dict(") < source.index("**_overrides,"), (
        "_overrides is built after the call that uses it")
    for field in ("dataset_name=", "tile_dir=", "h5_path=", "raw_dir="):
        assert field in source[source.index("_overrides = dict("):][:400], (
            f"_overrides does not carry {field}")


def test_the_registration_step_has_no_undefined_local_names(_tmp=None):
    """The general version of the check above, over the whole render function:
    every name it reads is either assigned in it, a parameter, or a module-level
    import. app_v28 cannot be imported here (it runs Streamlit at module scope),
    so this is the only way to catch a name that does not exist."""
    import ast

    source = APP.read_text()
    tree = ast.parse(source)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name == "_render_registration_step")

    assigned = {t.id for node in ast.walk(fn) for t in ast.walk(node)
                if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)}
    assigned |= {a.arg for a in fn.args.args}
    # `except X as e` binds through handler.name, a plain string, not a Name
    # node — so without this the collector reports every caught exception as
    # undefined. Same for `with ... as` targets already covered by Name/Store.
    assigned |= {h.name for h in ast.walk(fn)
                 if isinstance(h, ast.ExceptHandler) and h.name}
    module_level = {t.id for node in tree.body for t in ast.walk(node)
                    if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)}
    module_level |= {a.asname or a.name.split(".")[0]
                     for node in ast.walk(tree)
                     if isinstance(node, (ast.Import, ast.ImportFrom))
                     for a in node.names}
    module_level |= {n.name for n in tree.body
                     if isinstance(n, (ast.FunctionDef, ast.ClassDef))}

    import builtins
    known = assigned | module_level | set(dir(builtins))
    unknown = sorted({t.id for t in ast.walk(fn)
                      if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Load)
                      and t.id not in known})
    assert not unknown, f"names used but never defined: {unknown}"


def test_the_preview_reports_what_it_resolved(_tmp=None):
    """An override is a chance to register the wrong directory. The resolved
    values and where each came from have to be visible before the write."""
    server = SERVER_SOURCE
    assert '"resolved": plan.get("resolved")' in server
    assert '"sources": plan.get("sources")' in server
    assert 'st.table(' in APP.read_text()


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_kbtarget_test_"))
        try:
            fn(tmp_path)
            print(f"PASS  {name}")
        except Exception as e:
            failures.append(name)
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
