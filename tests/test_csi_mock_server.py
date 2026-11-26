"""Stdlib HTTP integration checks; these do not run CSI/model experiments."""

import copy
import http.client
import json
import math
import threading
import unittest

from tools.csi_mock_server import (FIXTURE_DIR, LINK_IDS, MODEL_IDS, MockHTTPServer,
                                   MockState, load_scenario, make_replay)


class MockHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = MockHTTPServer(("127.0.0.1", 0), enable_control=True)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=3)

    def setUp(self):
        self.server.mock_state = MockState()
        self.control({"freeze": True})

    def request(self, path, data=None, method=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        body = None if data is None else json.dumps(data)
        connection.request(method or ("GET" if data is None else "POST"), path,
                           body=body, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        status, headers, body = response.status, dict(response.getheaders()), response.read()
        connection.close()
        return status, headers, body

    def get(self, path):
        status, headers, body = self.request(path)
        return status, headers, json.loads(body) if body else None

    def control(self, data):
        status, _, body = self.request("/control", data)
        self.assertEqual(status, 200, body)
        return json.loads(body)

    def assert_prediction(self, row):
        self.assertEqual(row["source"], "mock")
        self.assertTrue(row["synthetic"])
        self.assertEqual(len(row["probs"]), 15)
        self.assertAlmostEqual(sum(row["probs"]), 1)
        self.assertEqual(row["label"], max(range(15), key=row["probs"].__getitem__))
        self.assertEqual(row["confidence"], row["probs"][row["label"]])
        self.assertEqual(row["x"], row["label"] % 5)
        self.assertEqual(row["y"], row["label"] // 5)
        self.assertAlmostEqual(row["x_m"], row["x"] * 0.8)
        self.assertAlmostEqual(row["y_m"], row["y"] * 0.8)
        self.assertEqual(row["window"]["end"], row["timestamp"])

    def test_meta_real_contract_grid_with_explicit_mock_capability(self):
        code, headers, meta = self.get("/api/meta")
        self.assertEqual(code, 200)
        self.assertEqual(headers["X-CSI-Source"], "mock")
        self.assertEqual(meta["contract_version"], "0.1")
        self.assertEqual(meta["source"], "mock")
        self.assertEqual(meta["supported_models"], list(MODEL_IDS))
        self.assertEqual(meta["grid"], {"cols": 5, "rows": 3, "spacing_m": 0.8})
        self.assertEqual(len(meta["cells"]), 15)
        self.assertEqual(meta["cells"][14]["x_m"], 3.2)
        self.assertEqual(meta["cells"][14]["y_m"], 1.6)
        self.assertEqual([i["id"] for i in meta["links"]], list(LINK_IDS))
        self.assertFalse(meta["mock_websocket_supported"])

    def test_default_envelope_contains_atomic_same_window_four_models(self):
        self.control({"advance": 5})
        code, _, batch = self.get("/api/predict/latest")
        self.assertEqual(code, 200)
        self.assertEqual({r["model_id"] for r in batch["predictions"]}, set(MODEL_IDS))
        for row in batch["predictions"]:
            self.assert_prediction(row)
            for field in ("session_id", "window_id", "seq", "timestamp"):
                self.assertEqual(row[field], batch[field])
            self.assertNotIn("ground_truth_label", row)
            self.assertNotIn("truth_source", row)

    def test_model_projection_and_unknown_model(self):
        code, _, row = self.get("/api/predict/latest?model_id=GCN")
        self.assertEqual(code, 200)
        self.assertEqual(row["model_id"], "GCN")
        self.assert_prediction(row)
        self.assertEqual(self.get("/api/predict/latest?model_id=GAT")[0], 400)

    def test_v01_fallback_does_not_claim_four_models(self):
        self.control({"multi_model": False})
        meta = self.get("/api/meta")[2]
        self.assertNotIn("supported_models", meta)
        row = self.get("/api/predict/latest")[2]
        self.assertEqual(row["type"], "prediction")
        for field in ("model_id", "window_id", "session_id", "predictions"):
            self.assertNotIn(field, row)
        self.assert_prediction(row)

    def test_no_predictions_204_while_not_ready(self):
        for state in ("offline", "calibrating", "training"):
            with self.subTest(state=state):
                self.control({"state": state, "progress": 0.4})
                code, _, body = self.request("/api/predict/latest")
                self.assertEqual((code, body), (204, b""))
                status = self.get("/api/status")[2]
                self.assertEqual(status["state"], state)
                self.assertEqual(status["source"], "mock")
                self.assertIsNone(status["calibrated_at"])
                if state == "offline":
                    self.assertTrue(all(not i["online"] for i in status["links"].values()))
                else:
                    self.assertEqual(status["progress"], 0.4)

    def test_degraded_links_and_recovery(self):
        self.control({"state": "degraded", "offline_links": ["piB_apY"]})
        status = self.get("/api/status")[2]
        self.assertFalse(status["links"]["piB_apY"]["online"])
        self.assertEqual(status["links"]["piB_apY"]["fps"], 0)
        self.assertGreater(status["links"]["piB_apY"]["last_frame_age_s"], 2)
        rows = self.get("/api/predict/latest")[2]["predictions"]
        self.assertTrue(all(r["state"] == "degraded" and "piB_apY" not in r["links_used"] for r in rows))
        self.control({"state": "ready"})
        self.assertTrue(all(i["online"] for i in self.get("/api/status")[2]["links"].values()))

    def test_intentional_http_malformed_and_invalid_probability_failures(self):
        self.control({"fault": "http_error"})
        self.assertEqual(self.get("/api/predict/latest")[0], 503)
        self.control({"fault": "malformed_json"})
        code, _, body = self.request("/api/predict/latest")
        self.assertEqual(code, 200)
        with self.assertRaises(json.JSONDecodeError):
            json.loads(body)
        self.control({"fault": "invalid_probs"})
        rows = self.get("/api/predict/latest")[2]["predictions"]
        self.assertTrue(all(sum(r["probs"]) > 1 for r in rows))
        self.control({"fault": "nonfinite"})
        rows = self.get("/api/predict/latest")[2]["predictions"]
        self.assertTrue(all(math.isnan(r["probs"][0]) for r in rows))

    def test_out_of_order_and_restart_are_observable(self):
        self.control({"advance": 10})
        before = self.get("/api/predict/latest")[2]
        self.control({"fault": "out_of_order"})
        older = self.get("/api/predict/latest")[2]
        self.assertLess(older["seq"], before["seq"])
        self.assertLess(older["timestamp"], before["timestamp"])
        self.assertEqual(older["session_id"], before["session_id"])
        self.control({"restart": True, "fault": "none"})
        restarted = self.get("/api/predict/latest")[2]
        self.assertEqual(restarted["seq"], 0)
        self.assertNotEqual(restarted["session_id"], before["session_id"])

    def test_stale_window_retains_identity_and_timestamp_contract(self):
        fresh = self.get("/api/predict/latest")[2]
        self.control({"fault": "stale"})
        stale = self.get("/api/predict/latest")[2]
        self.assertEqual(stale["window_id"], fresh["window_id"])
        self.assertLess(stale["timestamp"], fresh["timestamp"])
        self.assertEqual(stale["predictions"][0]["window"]["end"], stale["timestamp"])

    def test_replay_truth_is_predeclared_independently_and_synthetic(self):
        code, _, replay = self.get("/api/replay")
        self.assertEqual(code, 200)
        self.assertEqual(self.get("/replay.json")[2], replay)
        self.assertEqual(replay["source"], "mock")
        scenario = load_scenario()
        self.assertEqual(len(replay["predictions"]), len(scenario["frames"]) * 4)
        saw_error = False
        for index, row in enumerate(replay["predictions"]):
            self.assert_prediction(row)
            self.assertEqual(row["truth_source"], "fixture_label")
            self.assertEqual(row["ground_truth_label"], scenario["frames"][index // 4]["ground_truth_label"])
            saw_error |= row["label"] != row["ground_truth_label"]
        self.assertTrue(saw_error, "Fixtures must exercise a wrong prediction")
        changed = copy.deepcopy(scenario)
        changed["frames"][0]["predicted_labels"]["KNN"] = 14
        self.assertEqual(make_replay(changed)["predictions"][0]["ground_truth_label"], 0)

    def test_committed_replay_fixture_matches_served_replay(self):
        expected = json.loads((FIXTURE_DIR / "replay.json").read_text(encoding="utf-8"))
        self.assertEqual(expected, make_replay())

    def test_controls_validate_before_mutating_and_can_be_disabled(self):
        for invalid in ({"state": "fake"}, {"advance": -1}, {"progress": 1.1},
                        {"freeze": "yes"}, {"offline_links": ["fake"]}, {"unknown": True}):
            with self.subTest(invalid=invalid):
                self.assertEqual(self.request("/control", invalid)[0], 400)
        self.assertEqual(self.get("/api/status")[2]["state"], "ready")
        self.server.enable_control = False
        try:
            self.assertEqual(self.request("/control", {"state": "offline"})[0], 404)
        finally:
            self.server.enable_control = True

    def test_unknown_endpoint_websocket_and_cors_preflight(self):
        self.assertEqual(self.get("/not-a-real-endpoint")[0], 404)
        self.assertEqual(self.get("/ws/predictions")[0], 501)
        code, headers, body = self.request("/api/status", method="OPTIONS")
        self.assertEqual((code, body), (204, b""))
        self.assertEqual(headers["Access-Control-Allow-Origin"], "*")


if __name__ == "__main__":
    unittest.main()
