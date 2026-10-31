# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import datetime
import hashlib
import importlib
import json
import zipfile
from pathlib import Path

import pytest

OPENING = datetime.datetime(2026, 10, 23, 16, tzinfo=datetime.timezone.utc)


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "tools")
    )
    return importlib.import_module("flagos_submit_plan")


def item(tmp_path, operator="fp8_quantize"):
    return {
        "operator": operator,
        "manifest": str(tmp_path / "manifest.json"),
        "archive": str(tmp_path / (operator + ".zip")),
    }


def state(remaining=1, retry=0, submissions=None):
    return {
        "team_id": 2977,
        "quota": {"remaining": remaining, "retry_after_seconds": retry},
        "submissions": submissions or [],
    }


def forbidden(*args):
    pytest.fail("An external action was attempted")


def test_before_window_never_contacts_platform(runner, tmp_path):
    result = runner.execute(
        [item(tmp_path)],
        OPENING,
        "page",
        reader=forbidden,
        sender=forbidden,
        now=OPENING - datetime.timedelta(seconds=1),
    )
    assert result["status"] == "before_submission_window"


def test_exhausted_quota_never_uploads(runner, tmp_path):
    result = runner.execute(
        [item(tmp_path)],
        OPENING,
        "page",
        reader=lambda page: state(remaining=0),
        sender=forbidden,
        now=OPENING,
    )
    assert result["status"] == "quota_exhausted"


def test_completed_archive_is_skipped_even_without_quota(runner, tmp_path):
    existing = [
        {
            "operator": "fp8_quantize",
            "file_name": "fp8_quantize.zip",
            "submission_id": 123,
        }
    ]
    result = runner.execute(
        [item(tmp_path)],
        OPENING,
        "page",
        reader=lambda page: state(remaining=0, submissions=existing),
        sender=forbidden,
        now=OPENING,
    )
    assert result["status"] == "completed"
    assert result["results"][0]["submission_ids"] == [123]


def test_unconfirmed_submission_stops_queue(runner, tmp_path):
    sent = []

    def sender(page, manifest):
        sent.append(manifest)
        return {"accepted": [], "response": {"status": 503}}

    result = runner.execute(
        [item(tmp_path), item(tmp_path, "awq_dequantize")],
        OPENING,
        "page",
        reader=lambda page: state(),
        sender=sender,
        now=OPENING,
    )
    assert result["status"] == "submission_not_confirmed"
    assert len(sent) == 1
    assert (tmp_path / "receipt.json").exists()


def test_interval_is_respected_before_upload(runner, tmp_path):
    snapshots = iter([state(retry=120), state()])
    sleeps = []
    result = runner.execute(
        [item(tmp_path)],
        OPENING,
        "page",
        reader=lambda page: next(snapshots),
        sender=lambda page, path: {"accepted": [{"submission_id": 456}]},
        sleeper=sleeps.append,
        now=OPENING,
    )
    assert sleeps == [30]
    assert result["status"] == "completed"


def test_modified_archive_is_rejected(runner, tmp_path):
    archive = tmp_path / "fp8_quantize.zip"
    source = b"def solution(x): return x\n"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("fp8_quantize.py", source)
    report = tmp_path / "validation.xml"
    report.write_text(
        '<testsuites><testsuite tests="1" failures="0"/></testsuites>'
    )
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = {
        "operator": "fp8_quantize",
        "archive": archive.name,
        "zip_sha256": digest,
        "sources": {"fp8_quantize.py": hashlib.sha256(source).hexdigest()},
        "test_report": str(report),
        "test_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    plan = tmp_path / "plan.json"
    entry = dict(
        item(tmp_path),
        sha256=digest,
        priority=1,
        ready_for_official_evaluation=True,
    )
    plan.write_text(
        json.dumps(
            {
                "next_submissions": [entry],
                "next_quota_reset_kst": OPENING.isoformat(),
            }
        )
    )
    runner.verify_plan(plan)
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Archive checksum mismatch"):
        runner.verify_plan(plan)
