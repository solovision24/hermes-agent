from hermes_cli.adaptive_routing import classify_message, resolve_route


def test_classification_covers_coding_ladder_and_high_risk_override():
    assert classify_message("hello")[:2] == ("simple_chat", "simple")
    assert classify_message("write a small script")[:2] == ("simple_coding", "simple")
    assert classify_message("implement an API endpoint")[:2] == ("standard_coding", "standard")
    assert classify_message("debug a race condition across multiple files")[:2] == ("complex_coding", "complex")
    assert classify_message("design a database migration")[:2] == ("architecture_coding", "complex")


def test_disabled_and_missing_routes_fail_safe_to_current_target():
    assert resolve_route({}, "hello", current_model="m", current_provider="p").enabled is False
    route = resolve_route({"adaptive_model_routing": {"enabled": True, "routes": {}}},
                          "hello", current_model="m", current_provider="p")
    assert route.model == "m"
    assert route.provider == "p"


def test_policy_resolution_returns_explicit_target_and_reason():
    route = resolve_route({"adaptive_model_routing": {"enabled": True, "routes": {
        "simple_chat": {"provider": "cheap", "model": "small"}
    }}}, "hello")
    assert route.as_dict() == {"category": "simple_chat", "level": "simple",
                               "model": "small", "provider": "cheap",
                               "reason": "default conversational route", "enabled": True}


def test_category_can_define_level_specific_targets():
    route = resolve_route({"adaptive_model_routing": {"enabled": True, "routes": {
        "ops": {"levels": {"standard": {"model": "ops-small"},
                             "complex": {"model": "ops-large"}}}
    }}}, "deploy this service")
    assert route.model == "ops-small"
