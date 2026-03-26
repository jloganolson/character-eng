from character_eng.archive_analysis import ArchiveIssue, find_archive_state_issues


def test_find_archive_state_issues_flags_unsupported_set_name_and_summary_mismatch():
    events = [
        {
            "seq": 34,
            "type": "vision_state_update",
            "data": {
                "summary": "Updated person description for Greg.",
                "task_answers": [
                    {
                        "target_identity": "Person 2",
                        "answer": "The person is wearing glasses.",
                    }
                ],
                "person_updates": [
                    {
                        "person_id": "p1",
                        "set_name": "Greg",
                        "add_facts": ["The person is wearing glasses."],
                    }
                ],
                "add_facts": [],
            },
        }
    ]

    issues = find_archive_state_issues(events)

    assert ArchiveIssue(34, "vision_state_update", "unsupported_set_name", "p1 -> Greg") in issues
    assert ArchiveIssue(34, "vision_state_update", "summary_name_mismatch", "Updated person description for Greg.") in issues


def test_find_archive_state_issues_flags_speculative_facts():
    events = [
        {
            "seq": 145,
            "type": "vision_state_update",
            "data": {
                "summary": "Updated world state.",
                "task_answers": [{"target_identity": "", "answer": "The room appears to be a living space."}],
                "person_updates": [
                    {
                        "person_id": "p1",
                        "add_facts": [
                            "The person appears to be standing upright with their body slightly angled.",
                            "They might be reacting to something off-camera.",
                        ],
                    }
                ],
                "add_facts": [
                    "The room appears to be a combination of a living space and storage area.",
                ],
            },
        },
        {
            "seq": 231,
            "type": "reconcile",
            "data": {
                "add_facts": [
                    "The room could be part of a larger apartment.",
                ]
            },
        },
    ]

    issues = find_archive_state_issues(events)

    reasons = {(item.seq, item.reason) for item in issues}
    assert (145, "speculative_person_fact") in reasons
    assert (145, "speculative_world_fact") in reasons
    assert (231, "speculative_reconcile_fact") in reasons
