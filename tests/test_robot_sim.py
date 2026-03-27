import time

from character_eng.robot_sim import RobotSimController


def test_robot_sim_updates_face_state():
    robot = RobotSimController()

    state = robot.set_face(gaze="person", gaze_type="glance", expression="curious", source="test")

    assert state["face"]["gaze"] == "person"
    assert state["face"]["gaze_type"] == "glance"
    assert state["face"]["expression"] == "curious"
    assert state["face"]["source"] == "test"


def test_robot_sim_fistbump_contact_flow():
    robot = RobotSimController(fistbump_result_hold_s=0.05)

    offered = robot.offer_fistbump(source="test")
    session_id = offered["fistbump"]["session_id"]
    assert offered["fistbump"]["state"] == "offered"
    assert offered["fistbump"]["can_bump"] is True

    contacted = robot.register_fistbump_contact(session_id=session_id, source="test")
    assert contacted["fistbump"]["state"] == "contacted"
    assert contacted["fistbump"]["active"] is False

    time.sleep(0.07)
    settled = robot.snapshot()
    assert settled["fistbump"]["state"] == "idle"


def test_robot_sim_fistbump_timeout_expires_automatically():
    robot = RobotSimController(fistbump_timeout_s=0.05, fistbump_result_hold_s=0.05)

    robot.offer_fistbump(source="test")
    time.sleep(0.07)
    timed_out = robot.snapshot()
    assert timed_out["fistbump"]["state"] == "timed_out"

    time.sleep(0.07)
    settled = robot.snapshot()
    assert settled["fistbump"]["state"] == "idle"
