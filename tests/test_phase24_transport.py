"""Offline transport classification coverage for Phase 2.4."""

from email.utils import formatdate
import errno
import io
import unittest
from unittest.mock import patch
import urllib.error

from autoplay_qa.config import Settings
from autoplay_qa.nebius import NebiusClient, NebiusError, TransientNebiusError


class Phase24TransportTests(unittest.TestCase):
    def setUp(self):
        self.client = NebiusClient(Settings("unit-test-key"))

    def assert_transient(
        self,
        failure,
        reason,
        *,
        retry_after_seconds=None,
        possibly_billed=False,
    ):
        with patch.object(self.client._opener, "open", side_effect=failure) as transport:
            with self.assertRaises(TransientNebiusError) as context:
                self.client.list_models()
        error = context.exception
        self.assertIsInstance(error, NebiusError)
        self.assertEqual(error.reason, reason)
        self.assertEqual(error.retry_after_seconds, retry_after_seconds)
        self.assertIs(error.possibly_billed, possibly_billed)
        self.assertEqual(transport.call_count, 1)
        self.assertNotIn("unit-test-key", str(error))
        return error

    def test_direct_and_wrapped_timeouts_are_transient_and_possibly_billed(self):
        for failure in (
            TimeoutError("secret timeout detail"),
            urllib.error.URLError(TimeoutError("secret wrapped timeout detail")),
            OSError(errno.ETIMEDOUT, "secret errno timeout detail"),
        ):
            with self.subTest(failure=type(failure).__name__):
                error = self.assert_transient(
                    failure,
                    "timeout",
                    possibly_billed=True,
                )
                self.assertNotIn("secret", str(error))

    def test_direct_wrapped_and_errno_connection_resets_are_transient(self):
        for failure in (
            ConnectionResetError("secret reset detail"),
            urllib.error.URLError(ConnectionResetError("secret wrapped reset detail")),
            OSError(errno.ECONNRESET, "secret errno reset detail"),
            urllib.error.URLError(OSError(errno.ECONNRESET, "secret wrapped errno reset detail")),
        ):
            with self.subTest(failure=type(failure).__name__):
                error = self.assert_transient(failure, "connection_reset")
                self.assertNotIn("secret", str(error))

    def test_http_429_parses_delta_retry_after(self):
        failure = urllib.error.HTTPError(
            "url",
            429,
            "secret provider detail",
            {"Retry-After": "2.5"},
            io.BytesIO(b"unit-test-key secret provider body"),
        )
        error = self.assert_transient(
            failure,
            "rate_limited",
            retry_after_seconds=2.5,
        )
        self.assertNotIn("secret", str(error))

    def test_http_429_parses_http_date_and_ignores_invalid_header(self):
        now = 1_700_000_000.0
        date_failure = urllib.error.HTTPError(
            "url",
            429,
            "error",
            {"Retry-After": formatdate(now + 12, usegmt=True)},
            io.BytesIO(),
        )
        with patch("autoplay_qa.nebius.time.time", return_value=now):
            self.assert_transient(
                date_failure,
                "rate_limited",
                retry_after_seconds=12.0,
            )

        invalid_failure = urllib.error.HTTPError(
            "url",
            429,
            "error",
            {"Retry-After": "not a delay or date"},
            io.BytesIO(),
        )
        self.assert_transient(invalid_failure, "rate_limited")

    def test_every_http_5xx_is_transient_without_retry_after(self):
        for status in (500, 502, 503, 599):
            failure = urllib.error.HTTPError(
                "url",
                status,
                "secret provider detail",
                {"Retry-After": "9"},
                io.BytesIO(b"unit-test-key secret provider body"),
            )
            with self.subTest(status=status):
                error = self.assert_transient(failure, "http_5xx")
                self.assertIn(str(status), str(error))
                self.assertNotIn("secret", str(error))

    def test_dns_and_nontransient_http_status_remain_plain_nebius_errors(self):
        failures = (
            urllib.error.URLError("secret DNS detail"),
            urllib.error.HTTPError(
                "url",
                401,
                "secret provider detail",
                {},
                io.BytesIO(b"unit-test-key secret provider body"),
            ),
        )
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                with patch.object(self.client._opener, "open", side_effect=failure):
                    with self.assertRaises(NebiusError) as context:
                        self.client.list_models()
                self.assertNotIsInstance(context.exception, TransientNebiusError)
                self.assertNotIn("unit-test-key", str(context.exception))
                self.assertNotIn("secret", str(context.exception))

    def test_exception_attributes_reject_invalid_values(self):
        with self.assertRaises(ValueError):
            TransientNebiusError("bad", reason="other")
        for value in (-1, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                TransientNebiusError(
                    "bad",
                    reason="rate_limited",
                    retry_after_seconds=value,
                )


if __name__ == "__main__":
    unittest.main()
