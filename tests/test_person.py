from character_eng.person import Person, PersonUpdate, PeopleState


# --- Person ---


def test_person_add_fact_returns_scoped_id():
    p = Person(person_id="p1")
    fid1 = p.add_fact("Wearing a hat")
    fid2 = p.add_fact("Tall")
    assert fid1 == "p1f1"
    assert fid2 == "p1f2"
    assert p.facts == {"p1f1": "Wearing a hat", "p1f2": "Tall"}


def test_person_add_fact_ids_not_reused_after_removal():
    p = Person(person_id="p2")
    p.add_fact("A")
    p.add_fact("B")
    p.facts.pop("p2f1")
    fid3 = p.add_fact("C")
    assert fid3 == "p2f3"
    assert "p2f1" not in p.facts


def test_person_render():
    p = Person(person_id="p1", name="Alice", presence="present")
    p.add_fact("Wearing a red hat")
    text = p.render()
    assert "Alice (present)" in text
    assert "- Wearing a red hat" in text


def test_person_render_no_name():
    p = Person(person_id="p1", presence="approaching")
    text = p.render()
    assert "p1 (approaching)" in text


def test_person_render_for_reconcile():
    p = Person(person_id="p1", name="Alice", presence="present")
    p.add_fact("Wearing a hat")
    p.add_fact("Holding a book")
    text = p.render_for_reconcile()
    assert "p1 [Alice] (present):" in text
    assert "p1f1. Wearing a hat" in text
    assert "p1f2. Holding a book" in text


def test_person_render_for_reconcile_no_name():
    p = Person(person_id="p1", presence="approaching")
    text = p.render_for_reconcile()
    assert "p1 [p1] (approaching):" in text


# --- PeopleState ---


def test_people_state_add_person():
    ps = PeopleState()
    p1 = ps.add_person(name="Alice", presence="present")
    p2 = ps.add_person()
    assert p1.person_id == "p1"
    assert p1.name == "Alice"
    assert p2.person_id == "p2"
    assert p2.name is None
    assert len(ps.people) == 2


def test_people_state_find_by_name():
    ps = PeopleState()
    ps.add_person(name="Alice")
    ps.add_person(name="Bob")
    assert ps.find_by_name("alice").person_id == "p1"
    assert ps.find_by_name("BOB").person_id == "p2"
    assert ps.find_by_name("Charlie") is None


def test_people_state_present_people():
    ps = PeopleState()
    ps.add_person(name="Alice", presence="present")
    ps.add_person(name="Bob", presence="gone")
    ps.add_person(name="Charlie", presence="approaching")
    present = ps.present_people()
    assert len(present) == 2
    assert {p.name for p in present} == {"Alice", "Charlie"}


def test_people_state_render_empty():
    ps = PeopleState()
    assert ps.render() == ""


def test_people_state_render():
    ps = PeopleState()
    p = ps.add_person(name="Alice", presence="present")
    p.add_fact("Wearing a hat")
    text = ps.render()
    assert "Alice (present)" in text
    assert "- Wearing a hat" in text


def test_people_state_render_skips_gone():
    ps = PeopleState()
    ps.add_person(name="Alice", presence="present")
    ps.add_person(name="Bob", presence="gone")
    text = ps.render()
    assert "Alice" in text
    assert "Bob" not in text


def test_people_state_render_for_reconcile():
    ps = PeopleState()
    p = ps.add_person(name="Alice", presence="present")
    p.add_fact("Has a dog")
    text = ps.render_for_reconcile()
    assert "p1 [Alice] (present):" in text
    assert "p1f1. Has a dog" in text


def test_people_state_render_for_reconcile_empty():
    ps = PeopleState()
    assert ps.render_for_reconcile() == ""


def test_people_state_apply_updates_add_facts():
    ps = PeopleState()
    p = ps.add_person(name="Alice")
    updates = [PersonUpdate(person_id="p1", add_facts=["Wearing a hat", "Tall"])]
    ps.apply_updates(updates)
    assert list(p.facts.values()) == ["Wearing a hat", "Tall"]


def test_people_state_apply_updates_remove_facts():
    ps = PeopleState()
    p = ps.add_person(name="Alice")
    p.add_fact("Wearing a hat")
    p.add_fact("Tall")
    updates = [PersonUpdate(person_id="p1", remove_facts=["p1f1"])]
    ps.apply_updates(updates)
    assert list(p.facts.values()) == ["Tall"]


def test_people_state_apply_updates_set_name():
    ps = PeopleState()
    ps.add_person()
    updates = [PersonUpdate(person_id="p1", set_name="Alice")]
    ps.apply_updates(updates)
    assert ps.people["p1"].name == "Alice"


def test_people_state_apply_updates_set_presence():
    ps = PeopleState()
    ps.add_person(name="Alice", presence="approaching")
    updates = [PersonUpdate(person_id="p1", set_presence="present")]
    ps.apply_updates(updates)
    assert ps.people["p1"].presence == "present"


def test_people_state_apply_updates_invalid_presence_is_recorded():
    ps = PeopleState()
    ps.add_person(name="Alice", presence="approaching")
    updates = [PersonUpdate(person_id="p1", invalid_presence="teleporting")]
    ps.apply_updates(updates)
    assert ps.people["p1"].presence == "approaching"
    assert ps.people["p1"].history == ["ignored invalid presence: teleporting"]


def test_people_state_apply_updates_unknown_person_ignored():
    ps = PeopleState()
    ps.add_person(name="Alice")
    updates = [PersonUpdate(person_id="p99", add_facts=["Ghost"])]
    ps.apply_updates(updates)  # should not raise
    assert len(ps.people) == 1


def test_people_state_apply_updates_combined():
    ps = PeopleState()
    p = ps.add_person(presence="approaching")
    p.add_fact("Unknown person")
    updates = [PersonUpdate(
        person_id="p1",
        remove_facts=["p1f1"],
        add_facts=["Wearing a blue jacket"],
        set_name="Dave",
        set_presence="present",
    )]
    ps.apply_updates(updates)
    assert p.name == "Dave"
    assert p.presence == "present"
    assert "p1f1" not in p.facts
    assert "Wearing a blue jacket" in list(p.facts.values())


def test_people_state_apply_updates_invalid_fact_id_ignored():
    ps = PeopleState()
    p = ps.add_person(name="Alice")
    p.add_fact("A fact")
    updates = [PersonUpdate(person_id="p1", remove_facts=["p1f99"])]
    ps.apply_updates(updates)
    assert list(p.facts.values()) == ["A fact"]


def test_people_state_apply_updates_replaces_ephemeral_facts():
    ps = PeopleState()
    p = ps.add_person(name="Alice")
    ps.apply_updates([PersonUpdate(person_id="p1", add_facts=["Looking left"], fact_scope="ephemeral")])
    first_id = next(iter(p.facts))

    ps.apply_updates([PersonUpdate(person_id="p1", add_facts=["Looking right"], fact_scope="ephemeral")])

    assert list(p.facts.values()) == ["Looking right"]
    assert first_id not in p.facts
    only_id = next(iter(p.facts))
    assert p.fact_scope(only_id) == "ephemeral"


def test_people_state_apply_updates_upgrades_matching_fact_to_static():
    ps = PeopleState()
    p = ps.add_person(name="Alice")
    ps.apply_updates([PersonUpdate(person_id="p1", add_facts=["Wearing glasses"], fact_scope="ephemeral")])
    fact_id = next(iter(p.facts))

    ps.apply_updates([PersonUpdate(person_id="p1", add_facts=["Wearing glasses"], fact_scope="static")])

    assert list(p.facts.values()) == ["Wearing glasses"]
    assert next(iter(p.facts)) == fact_id
    assert p.fact_scope(fact_id) == "static"


def test_people_state_show_panel():
    ps = PeopleState()
    ps.add_person(name="Alice", presence="present")
    panel = ps.show()
    assert panel.title == "People"


def test_people_state_show_empty():
    ps = PeopleState()
    panel = ps.show()
    assert panel.title == "People"
