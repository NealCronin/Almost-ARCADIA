"""
Tests for the persistent settings store — with real disk I/O and deep-copy verification.
"""

import json
import os
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch

from django.test import TestCase

from core_orchestrator.services.settings_store import (
    SettingsStore,
    SettingsValidationError,
    AppSettings,
    _dict_to_appsettings,
    _apply_legacy_migration,
    config_dir,
    reset_settings_store_for_tests,
)
from .test_helpers import setup_test_config_dir, teardown_test_config_dir, reset_all_singletons


class SettingsStoreBaseTests(TestCase):
    def setUp(self):
        self.tmpdir = setup_test_config_dir()
        reset_all_singletons()

    def tearDown(self):
        teardown_test_config_dir(self.tmpdir)
        reset_all_singletons()

    def _store(self):
        return SettingsStore()

    def test_first_run_creates_defaults(self):
        store = self._store()
        settings = store.load()
        self.assertEqual(settings.version, 1)
        self.assertEqual(settings.client.llm_mode, "local")
        self.assertEqual(settings.client.sam3_mode, "local")
        self.assertTrue(store.file_path.exists())

    def test_save_and_reload(self):
        store = self._store()
        settings = store.load()
        settings.client.llm_mode = "remote"
        settings.client.sam3_mode = "remote"
        settings.host.listen_ip = "192.168.1.100"
        store.save(settings)

        store2 = SettingsStore()
        loaded = store2.load()
        self.assertEqual(loaded.client.llm_mode, "remote")
        self.assertEqual(loaded.client.sam3_mode, "remote")
        self.assertEqual(loaded.host.listen_ip, "192.168.1.100")

    def test_deep_copy_load_prevents_cache_mutation(self):
        """Mutating the returned object must not affect the internal cache."""
        store = self._store()
        s1 = store.load()
        s1.client.llm_mode = "remote"
        # Load again — should still be local
        s2 = store.load()
        self.assertEqual(s2.client.llm_mode, "local")

    def test_deep_copy_save_prevents_caller_mutation(self):
        """After save, mutating the caller's object must not affect the cache."""
        store = self._store()
        s = store.load()
        s.client.llm_mode = "remote"
        store.save(s)
        # Mutate after save
        s.client.llm_mode = "local"
        # Load again — should be remote
        s2 = store.load()
        self.assertEqual(s2.client.llm_mode, "remote")

    def test_update_merges_correctly(self):
        store = self._store()
        store.load()
        result = store.update({"client": {"sam3_mode": "remote"}})
        self.assertEqual(result.client.llm_mode, "local")
        self.assertEqual(result.client.sam3_mode, "remote")

    def test_reset_to_defaults(self):
        store = self._store()
        store.update({"client": {"llm_mode": "remote", "sam3_mode": "remote"}})
        defaults = store.reset_to_defaults()
        self.assertEqual(defaults.client.llm_mode, "local")
        self.assertEqual(defaults.client.sam3_mode, "local")

    def test_malformed_json_recovery_with_new_store(self):
        """Malformed JSON is recovered from backup using a NEW store instance."""
        store = self._store()
        store.load()
        s = store.load()
        s.client.llm_mode = "remote"
        store.save(s)
        # Save again to create backup
        store.save(s)

        # Corrupt the primary file
        store.file_path.write_text("{invalid json!!!", encoding="utf-8")

        # Create a NEW store — must not use the old cache
        store2 = SettingsStore()
        loaded = store2.load()
        self.assertEqual(loaded.client.llm_mode, "remote")

        # Verify primary file was restored from backup
        self.assertTrue(store.file_path.exists())
        raw = store.file_path.read_text()
        data = json.loads(raw)
        self.assertEqual(data["client"]["llm_mode"], "remote")

        # Verify a third store reads the restored primary
        store3 = SettingsStore()
        loaded3 = store3.load()
        self.assertEqual(loaded3.client.llm_mode, "remote")

    def test_schema_version_present(self):
        store = self._store()
        store.save(AppSettings())
        data = json.loads(store.file_path.read_text())
        self.assertEqual(data["version"], 1)

    def test_unknown_nested_fields_preserved(self):
        """Unknown nested fields survive round-trips."""
        store = self._store()
        store.update({"host": {"llm": {"future_backend_option": "value"}}})
        # Update an unrelated known field
        store.update({"client": {"dataset_path": "/data/video.mp4"}})

        # Read raw file
        raw = store.file_path.read_text()
        data = json.loads(raw)
        self.assertEqual(data["host"]["llm"]["future_backend_option"], "value")
        self.assertEqual(data["client"]["dataset_path"], "/data/video.mp4")

    def test_concurrent_reads_and_writes(self):
        """Concurrent read/write don't corrupt the file."""
        store = self._store()
        store.load()

        errors = []

        def reader():
            for _ in range(20):
                try:
                    s = store.load()
                    if not isinstance(s, AppSettings):
                        errors.append("Bad return type")
                except Exception as e:
                    errors.append(str(e))

        def writer():
            for i in range(20):
                try:
                    store.update({"client": {"dataset_path": f"/path/{i}"}})
                except Exception as e:
                    errors.append(str(e))

        threads = [threading.Thread(target=reader) for _ in range(3)]
        threads += [threading.Thread(target=writer) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])

        # Final file must be valid JSON
        raw = store.file_path.read_text()
        data = json.loads(raw)
        self.assertEqual(data["version"], 1)


class SettingsStoreValidationTests(TestCase):
    def setUp(self):
        self.tmpdir = setup_test_config_dir()
        reset_all_singletons()

    def tearDown(self):
        teardown_test_config_dir(self.tmpdir)
        reset_all_singletons()

    def test_invalid_port_raises_validation_error(self):
        store = SettingsStore()
        store.load()
        with self.assertRaises(SettingsValidationError):
            store.update({"host": {"llm": {"port": "abc"}}})

    def test_port_out_of_range(self):
        store = SettingsStore()
        store.load()
        with self.assertRaises(SettingsValidationError):
            store.update({"host": {"llm": {"port": 99999}}})

    def test_invalid_mode_value(self):
        store = SettingsStore()
        store.load()
        with self.assertRaises(SettingsValidationError):
            store.update({"client": {"llm_mode": "invalid_mode"}})

    def test_invalid_service_mode(self):
        store = SettingsStore()
        store.load()
        with self.assertRaises(SettingsValidationError):
            store.update({"host": {"llm": {"service_mode": "cloud"}}})

    def test_invalid_api_format(self):
        store = SettingsStore()
        store.load()
        with self.assertRaises(SettingsValidationError):
            store.update({"host": {"llm": {"api_format": "unsupported"}}})

    def test_arguments_must_be_list(self):
        store = SettingsStore()
        store.load()
        with self.assertRaises(SettingsValidationError):
            store.update({"host": {"llm": {"arguments": "--ctx-size 4096"}}})

    def test_arguments_items_must_be_strings(self):
        store = SettingsStore()
        store.load()
        with self.assertRaises(SettingsValidationError):
            store.update({"host": {"llm": {"arguments": [1, 2, 3]}}})

    def test_bool_string_false_becomes_false(self):
        """bool("false") must become False, not True."""
        store = SettingsStore()
        store.load()
        result = store.update({"host": {"llm": {"auto_start": "false"}}})
        self.assertFalse(result.host.llm.auto_start)

    def test_bool_arbitrary_string_rejected(self):
        """Arbitrary string bool values are rejected."""
        store = SettingsStore()
        store.load()
        with self.assertRaises(SettingsValidationError):
            store.update({"host": {"llm": {"auto_start": "maybe"}}})

    def test_scheme_validation(self):
        store = SettingsStore()
        store.load()
        with self.assertRaises(SettingsValidationError):
            store.update({"client": {"remote_host": {"scheme": "ftp"}}})


class SettingsStoreMigrationTests(TestCase):
    def test_legacy_routing_mode_local(self):
        migrated = _apply_legacy_migration({"routing_mode": "local", "version": 0})
        self.assertEqual(migrated["client"]["llm_mode"], "local")
        self.assertEqual(migrated["client"]["sam3_mode"], "local")

    def test_legacy_routing_mode_remote(self):
        migrated = _apply_legacy_migration({"routing_mode": "remote", "version": 0})
        self.assertEqual(migrated["client"]["llm_mode"], "remote")
        self.assertEqual(migrated["client"]["sam3_mode"], "remote")

    def test_new_schema_no_migration(self):
        data = {"version": 1, "client": {"llm_mode": "local", "sam3_mode": "remote"}}
        migrated = _apply_legacy_migration(data)
        self.assertEqual(migrated["client"]["llm_mode"], "local")
        self.assertEqual(migrated["client"]["sam3_mode"], "remote")
