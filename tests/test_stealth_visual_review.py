"""Fail-closed checks for the human visual-review gate."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/rls/supermimic/check_stealth_visual_review.py"
SPEC = importlib.util.spec_from_file_location("check_stealth_visual_review", SCRIPT)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


class VisualReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / "review"
        self.directory.mkdir()
        self.checkpoint = self.root / "checkpoint.pt"
        self.checkpoint.write_bytes(b"checkpoint")
        (self.directory / "trajectories.npz").write_bytes(b"trajectory")
        for name in ("front.mp4", "side.mp4", "commands.mp4"):
            (self.directory / name).write_bytes(name.encode())
        self.review_path = self.directory / "visual_review.json"
        self.output = self.root / "result.json"
        self.write_review()

    def write_review(self, **updates):
        scope = updates.get("review_scope", "full")
        criteria = gate.ENTRY_CRITERIA if scope == "entry" else gate.CRITERIA
        review = {
            "review_scope": scope,
            "passed": True,
            "reviewer": "human reviewer",
            "reviewed_at": "2026-11-08T12:00:00+08:00",
            "notes": "Inspected entry, gait, turns, and key release in the rendered clips.",
            "checkpoint_sha256": gate.sha256(self.checkpoint),
            "trajectory_sha256": gate.sha256(self.directory / "trajectories.npz"),
            "criteria": {name: True for name in criteria},
            "videos": [
                {"path": name, "sha256": gate.sha256(self.directory / name)}
                for name in ("front.mp4", "side.mp4", "commands.mp4")
            ],
            "camera_views": {
                "front": "front.mp4", "side": "side.mp4", "commands": "commands.mp4"
            },
        }
        review.update(updates)
        self.review_path.write_text(json.dumps(review), encoding="utf-8")

    def validate(self, scope="full"):
        return gate.validate(self.directory, self.checkpoint, scope)[0]

    def run_cli(self, output=None, scope="full"):
        with redirect_stdout(io.StringIO()):
            return gate.main([
                "--directory", str(self.directory), "--checkpoint", str(self.checkpoint),
                "--output", str(output or self.output), "--scope", scope,
            ])

    def test_complete_review_passes_and_binds_all_video_hashes(self):
        result = self.validate()
        self.assertTrue(result["passed"], result["reasons"])
        self.assertEqual(len(result["videos"]), 3)
        self.assertEqual(result["files"]["video:front.mp4"]["sha256"],
                         gate.sha256(self.directory / "front.mp4"))
        self.assertEqual(result["review_scope"], "full")
        self.assertTrue(result["files"]["checkpoint"]["matched"])

    def test_checkpoint_trajectory_and_video_tampering_each_fail(self):
        for filename, mutate in (
            (self.checkpoint, lambda: self.checkpoint.write_bytes(b"changed checkpoint")),
            (self.directory / "trajectories.npz",
             lambda: (self.directory / "trajectories.npz").write_bytes(b"changed trajectory")),
            (self.directory / "front.mp4",
             lambda: (self.directory / "front.mp4").write_bytes(b"changed video")),
        ):
            with self.subTest(filename=filename):
                original = filename.read_bytes()
                mutate()
                self.assertFalse(self.validate()["passed"])
                filename.write_bytes(original)

    def test_missing_or_false_criterion_fails(self):
        for name in gate.CRITERIA:
            for value in ("missing", False):
                with self.subTest(name=name, value=value):
                    criteria = dict.fromkeys(gate.CRITERIA, True)
                    if value == "missing":
                        del criteria[name]
                    else:
                        criteria[name] = value
                    self.write_review(criteria=criteria, numeric_passed=True,
                                      quality_gate={"passed": True})
                    self.assertFalse(self.validate()["passed"])

    def test_entry_review_cannot_qualify_full_locomotion(self):
        self.write_review(review_scope="entry")
        self.assertTrue(self.validate("entry")["passed"])
        full = self.validate("full")
        self.assertFalse(full["passed"])
        self.assertEqual(full["review_scope"], "entry")
        self.assertEqual(full["requested_scope"], "full")
        self.assertIn("review_scope must be exactly 'full'; scope cannot be promoted",
                      full["reasons"])

    def test_entry_scope_requires_both_views_but_not_command_view(self):
        self.write_review(review_scope="entry")
        review = json.loads(self.review_path.read_text())
        review["videos"] = [item for item in review["videos"] if item["path"] != "commands.mp4"]
        review["camera_views"].pop("commands")
        self.review_path.write_text(json.dumps(review))
        self.assertTrue(self.validate("entry")["passed"])
        review["camera_views"].pop("side")
        self.review_path.write_text(json.dumps(review))
        self.assertFalse(self.validate("entry")["passed"])

    def test_path_traversal_and_absolute_names_fail(self):
        for path in ("../outside.mp4", "a/../../outside.mp4", "/outside.mp4",
                     "D:/outside.mp4", "D:outside.mp4", "a\\front.mp4", "//host/a.mp4",
                     "./front.mp4", "a//front.mp4"):
            with self.subTest(path=path):
                self.write_review()
                review = json.loads(self.review_path.read_text())
                review["videos"][0]["path"] = path
                self.review_path.write_text(json.dumps(review))
                self.assertFalse(self.validate()["passed"])

    def test_video_symlink_escape_fails(self):
        outside = self.root / "outside.mp4"
        outside.write_bytes(b"outside")
        link = self.directory / "escape.mp4"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        review = json.loads(self.review_path.read_text())
        review["videos"][0] = {"path": "escape.mp4", "sha256": gate.sha256(outside)}
        review["camera_views"]["front"] = "escape.mp4"
        self.review_path.write_text(json.dumps(review))
        self.assertFalse(self.validate()["passed"])

    def test_output_cannot_overwrite_any_reviewed_input(self):
        for path in (self.checkpoint, self.review_path, self.directory / "trajectories.npz",
                     self.directory / "front.mp4", self.directory / "side.mp4",
                     self.directory / "commands.mp4"):
            with self.subTest(path=path):
                original = path.read_bytes()
                self.assertEqual(self.run_cli(output=path), 2)
                self.assertEqual(path.read_bytes(), original)

    def test_malformed_receipt_writes_failed_result_and_exits_two(self):
        self.review_path.write_text("{broken", encoding="utf-8")
        result = self.run_cli()
        self.assertEqual(result, 2)
        receipt = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertFalse(receipt["passed"])
        self.assertTrue(receipt["reasons"])

    def test_output_hardlink_to_input_is_rejected(self):
        try:
            self.output.hardlink_to(self.directory / "front.mp4")
        except OSError:
            self.skipTest("hardlinks unavailable")
        original = (self.directory / "front.mp4").read_bytes()
        result = self.run_cli()
        self.assertEqual(result, 2)
        self.assertEqual((self.directory / "front.mp4").read_bytes(), original)

    def test_scalar_quality_cannot_override_rejected_visual_review(self):
        self.write_review(passed=False, numeric_passed=True, quality_gate={"passed": True})
        result = self.validate()
        self.assertFalse(result["passed"])
        self.assertIn("review receipt passed must be explicitly true", result["reasons"])

    def test_exact_true_required_for_each_criterion(self):
        criteria = {name: True for name in gate.CRITERIA}
        criteria["arms_natural"] = 1
        self.write_review(criteria=criteria)
        self.assertFalse(self.validate()["passed"])

    def test_unknown_criterion_cannot_extend_approval_scope(self):
        criteria = dict.fromkeys(gate.CRITERIA, True)
        criteria["looks_good"] = True
        self.write_review(criteria=criteria)
        self.assertFalse(self.validate()["passed"])

    def test_review_identity_notes_and_timestamp_are_required(self):
        for field in ("reviewer", "notes", "reviewed_at"):
            for value in ("", " ", None, 1):
                with self.subTest(field=field, value=value):
                    self.write_review(**{field: value})
                    self.assertFalse(self.validate()["passed"])
        self.write_review(reviewed_at="yesterday")
        self.assertFalse(self.validate()["passed"])
        self.write_review(reviewed_at="2026-11-08T03:00:00Z")
        self.assertTrue(self.validate()["passed"])

    def test_malformed_field_types_fail_with_receipt_instead_of_crashing(self):
        cases = [
            {"videos": None}, {"videos": 1}, {"videos": {}}, {"videos": []},
            {"videos": [None]}, {"videos": [{"path": []}]},
            {"videos": [{"path": "front.mp4", "sha256": []}]},
            {"camera_views": None}, {"camera_views": {"front": []}},
            {"criteria": []}, {"review_scope": []}, {"passed": 1},
        ]
        for index, updates in enumerate(cases):
            with self.subTest(updates=updates):
                self.write_review(**updates)
                output = self.root / f"malformed-{index}.json"
                self.assertEqual(self.run_cli(output=output), 2)
                self.assertFalse(json.loads(output.read_text())["passed"])

    def test_video_count_is_bounded_and_duplicates_fail(self):
        review = json.loads(self.review_path.read_text())
        self.write_review(videos=review["videos"] * 11)
        self.assertFalse(self.validate()["passed"])
        self.write_review(videos=review["videos"] + review["videos"][:1])
        self.assertFalse(self.validate()["passed"])

    def test_relative_subdirectory_video_is_allowed(self):
        nested = self.directory / "camera"
        nested.mkdir()
        (nested / "front.mp4").write_bytes(b"nested front")
        review = json.loads(self.review_path.read_text())
        review["videos"][0] = {"path": "camera/front.mp4",
                               "sha256": gate.sha256(nested / "front.mp4")}
        review["camera_views"]["front"] = "camera/front.mp4"
        self.review_path.write_text(json.dumps(review))
        self.assertTrue(self.validate()["passed"])

    def test_checkpoint_symlink_can_point_outside_review_directory(self):
        link = self.root / "alias.pt"
        try:
            link.symlink_to(self.checkpoint)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        result, _ = gate.validate(self.directory, link, "full")
        self.assertTrue(result["passed"], result["reasons"])
        self.assertEqual(result["files"]["checkpoint"]["path"], str(self.checkpoint.resolve()))

    def test_missing_input_still_produces_failure_receipt(self):
        (self.directory / "trajectories.npz").unlink()
        self.assertEqual(self.run_cli(), 2)
        self.assertFalse(json.loads(self.output.read_text())["passed"])

    def test_duplicate_json_keys_are_rejected(self):
        raw = self.review_path.read_text()
        self.review_path.write_text('{"passed": false,' + raw[1:])
        self.assertFalse(self.validate()["passed"])

    def test_real_cli_defaults_full_and_returns_status_codes(self):
        command = [sys.executable, "-B", str(SCRIPT), "--directory", str(self.directory),
                   "--checkpoint", str(self.checkpoint), "--output", str(self.output)]
        full = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(full.returncode, 0, full.stderr + full.stdout)
        self.assertEqual(json.loads(full.stdout)["review_scope"], "full")
        self.write_review(review_scope="entry")
        command[-1] = str(self.root / "full-refusal.json")
        refusal = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(refusal.returncode, 2, refusal.stderr + refusal.stdout)
        self.assertFalse(json.loads(refusal.stdout)["passed"])


if __name__ == "__main__":
    unittest.main()
