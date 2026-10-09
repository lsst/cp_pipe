# This file is part of cp_pipe.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import contextlib
import glob
import json
import os
import re

import pytest
import vcr
import yaml

import lsst.utils.tests

from lsst.cp.pipe.utilsEfd import CpEfdClient

TESTDIR = os.path.abspath(os.path.dirname(__file__))
CASSETTE_DIR = os.path.join(TESTDIR, "data", "cassettes")
EMPTY_EFDAUTH = os.path.join(TESTDIR, "data", "efdauth_sanitized.json")

REDACTED = "REDACTED"
_PASSWORD_KEY = re.compile(r"pass(word|wd)", re.IGNORECASE)


def _redact(obj):
    """Recursively redact password-like values in place.

    Returns True if anything was changed.
    """
    changed = False
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(key, str) and _PASSWORD_KEY.search(key) and isinstance(value, str) and value:
                obj[key] = REDACTED
                changed = True
            else:
                changed |= _redact(value)
    elif isinstance(obj, list):
        for item in obj:
            changed |= _redact(item)
    return changed


def _scrubBody(body):
    """Return a redacted copy of a JSON body, or None if nothing changed."""
    if not body:
        return None
    try:
        data = json.loads(body)
    except (ValueError, UnicodeDecodeError, TypeError):
        return None
    if not _redact(data):
        return None
    text = json.dumps(data)
    return text.encode() if isinstance(body, bytes) else text


def _scrubRequest(request):
    new = _scrubBody(request.body)
    if new is not None:
        request.body = new
    return request


def _scrubResponse(response):
    body = response.get("body", {})
    new = _scrubBody(body.get("string"))
    if new is not None:
        body["string"] = new
        length = len(new if isinstance(new, bytes) else new.encode())
        headers = response.get("headers", {})
        for key in headers:
            if key.lower() == "content-length":
                headers[key] = [str(length)]
    return response


VCR_CONFIG = dict(
    match_on=["method", "scheme", "host", "port", "path", "query", "body"],
    decode_compressed_response=True,  # so response bodies are scrubbable JSON
    filter_headers=["authorization", "cookie"],  # basic auth carries the password
    filter_query_parameters=["p", "password"],  # influx-style ?u=...&p=...
    filter_post_data_parameters=["p", "password"],
    before_record_request=_scrubRequest,
    before_record_response=_scrubResponse,  # e.g. credential-service JSON
)


# pytest-recording hooks
@pytest.fixture(scope="module")
def vcr_config():
    return VCR_CONFIG


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    return CASSETTE_DIR


@pytest.fixture(scope="class")
def efdClient(request):
    """Class-level replacement for setUpClass.

    Playback (``--record-mode=none``, the default): EFDAUTH is set to an
    empty-credentials file in tests/data.
    Recording (any other mode, or --disable-recording): EFDAUTH is taken
    from the calling environment.
    """
    recordMode = request.config.getoption("--record-mode") or "none"
    disabled = request.config.getoption("--disable-recording")
    recording = disabled or recordMode != "none"

    with pytest.MonkeyPatch.context() as mp:
        if recording:
            if not os.environ.get("EFDAUTH"):
                pytest.fail("EFDAUTH must be set in the environment to record EFD cassettes.")
        else:
            mp.setenv("EFDAUTH", EMPTY_EFDAUTH)

        if disabled:
            ctx = contextlib.nullcontext()
        else:
            cassette = os.path.join(CASSETTE_DIR, f"{request.cls.__name__}.setUpClass.yaml")
            mode = recordMode
            if mode == "rewrite":  # pytest-recording-only mode; emulate for vcrpy
                if os.path.exists(cassette):
                    os.remove(cassette)
                mode = "all"
            ctx = vcr.VCR(record_mode=mode, **VCR_CONFIG).use_cassette(cassette)

        with ctx:
            try:
                request.cls.client = CpEfdClient()
            except Exception as e:
                if recording:
                    pytest.skip(f"Could not initialize EFD client: {e}")
                raise  # in playback, a failure here is a real regression

        yield  # EFDAUTH stays patched for all tests in the class


@pytest.mark.usefixtures("efdClient")
class UtilsEfdTestCase(lsst.utils.tests.TestCase):
    """Unit test for EFD access code."""

    @pytest.mark.vcr
    def test_monochromator(self):
        data = self.client.getEfdMonochromatorData(
            dateMin="2023-12-19T00:00:00",
            dateMax="2023-12-19T23:59:59"
        )

        indexDate, wavelength = self.client.parseMonochromatorStatus(
            data,
            "2023-12-19T14:37:19.498"
        )
        self.assertEqual(wavelength, 550.0)
        self.assertEqual(indexDate, "2023-12-19T14:37:17.799")

    @pytest.mark.vcr
    def test_electrometer(self):
        data = self.client.getEfdElectrometerData(
            dateMin="2024-05-30T00:00:00",
            dateMax="2024-05-30T05:00:00",
        )

        # Test single lookups:
        for (iDate, rDate), (iVal, rVal) in zip([("2024-05-30T04:21:48.6", "2024-05-30T04:22:08"),
                                                 ("2024-05-30T04:21:50", "2024-05-30T04:22:10"),
                                                 ("2024-05-30T04:21:53", "2024-05-30T04:22:13"),
                                                 ("2024-05-30T04:21:58", "2024-05-30T04:22:18"),
                                                 ("2024-05-30T04:22:17", "2024-05-30T04:22:37")],
                                                [(-1.5269e-07, -1.5244e-07),
                                                 (-1.5168e-07, -1.5137e-07),
                                                 (-1.5165e-07, -1.5205e-07),
                                                 (-1.5223e-07, -1.5147e-07),
                                                 (-1.5226e-07, -1.5558e-07)]):
            indexDate, intensity, _ = self.client.parseElectrometerStatus(
                data,
                iDate
            )
            self.assertFloatsAlmostEqual(intensity, iVal, atol=1e-10)

            indexDateReference, intensityReference, _ = self.client.parseElectrometerStatus(
                data,
                rDate
            )
            self.assertFloatsAlmostEqual(intensityReference, rVal, atol=1e-10)

        # Test integrated lookups:
        iDate = "2024-05-30T04:21:48.6"
        iDateEnd = "2024-05-30T04:22:18"
        rDate = "2024-05-30T04:22:08"
        rDateEnd = "2024-05-30T04:22:38"
        indexDate, intensity, endDate = self.client.parseElectrometerStatus(
            data,
            iDate,
            dateEnd=iDateEnd,
            doIntegrateSamples=True,
            index=201
        )
        self.assertFloatsAlmostEqual(intensity, -1.52297e-7, atol=1e-10)
        indexDateReference, intensityReference, endDateReference = self.client.parseElectrometerStatus(
            data,
            rDate,
            dateEnd=rDateEnd,
            doIntegrateSamples=True,
            index=201
        )
        self.assertFloatsAlmostEqual(intensityReference, -1.532977e-07, atol=1e-10)

    @pytest.mark.vcr
    def test_electrometer_alternate(self):
        # This should raise if no dates are passed:
        with self.assertRaises(RuntimeError):
            data = self.client.getEfdElectrometerData(
                dataSeries='lsst.sal.Electrometer.logevent_logMessage')

        data = self.client.getEfdElectrometerData(
            dataSeries='lsst.sal.Electrometer.logevent_logMessage',
            dateMin='2024-07-26T16:30:00',
            dateMax='2024-07-26T16:45:00')

        # Test single lookups.  These should not be integrated.
        for iDate, iVal in zip(["2024-07-26T16:38:32.228",
                                "2024-07-26T16:38:54.581",
                                "2024-07-26T16:40:56.579",
                                "2024-07-26T16:42:58.553"],
                               [-2.24234e-07,
                                -2.24388e-07,
                                -2.24105e-07,
                                -2.23784e-07]):
            indexDate, intensity, _ = self.client.parseElectrometerStatus(
                data,
                iDate
            )
            self.assertFloatsAlmostEqual(intensity, iVal, atol=1e-10)


class CassetteSanitizationTestCase(lsst.utils.tests.TestCase):
    """Guard against committing credentials again."""

    def test_no_credentials_in_cassettes(self):
        pattern = re.compile(r'"[^"]*pass(?:word|wd)[^"]*"\s*:\s*"([^"]*)"', re.IGNORECASE)
        for fname in glob.glob(os.path.join(CASSETTE_DIR, "*.yaml")):
            with open(fname) as f:
                cassette = yaml.safe_load(f)
            for interaction in cassette.get("interactions", []):
                req = interaction["request"]
                headers = {k.lower() for k in req.get("headers", {})}
                self.assertNotIn("authorization", headers, msg=fname)
                self.assertIsNone(re.search(r"[?&]p(assword)?=", req.get("uri", "")), msg=fname)
                body = interaction["response"]["body"].get("string") or ""
                if isinstance(body, bytes):
                    body = body.decode(errors="replace")
                for m in pattern.finditer(body):
                    self.assertIn(m.group(1), ("", REDACTED), msg=fname)
