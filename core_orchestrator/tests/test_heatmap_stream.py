"""
Tests for heatmap_stream endpoint.
"""
import os
import unittest
from unittest.mock import patch

from django.test import TestCase, RequestFactory
from django.http import StreamingHttpResponse

from core_orchestrator.views import heatmap_stream
from .test_helpers import setup_test_config_dir, teardown_test_config_dir, reset_all_singletons


class HeatmapStreamTests(TestCase):
    def setUp(self):
        self.tmpdir = setup_test_config_dir()
        reset_all_singletons()
        self.factory = RequestFactory()

    def tearDown(self):
        teardown_test_config_dir(self.tmpdir)
        reset_all_singletons()

    def test_returns_streaming_response(self):
        request = self.factory.get('/stream/heatmap/')
        response = heatmap_stream(request)
        self.assertIsInstance(response, StreamingHttpResponse)

    def test_with_local_mode(self):
        request = self.factory.get('/stream/heatmap/', {
            'routing_mode': 'local',
            'dataset_path': '',
        })
        response = heatmap_stream(request)
        self.assertIsInstance(response, StreamingHttpResponse)

    def test_with_remote_mode(self):
        request = self.factory.get('/stream/heatmap/', {
            'routing_mode': 'remote',
            'remote_host_ip': '127.0.0.1',
            'remote_host_port': '8080',
        })
        response = heatmap_stream(request)
        self.assertIsInstance(response, StreamingHttpResponse)

    def test_with_independent_modes(self):
        request = self.factory.get('/stream/heatmap/', {
            'llm_mode': 'local',
            'sam3_mode': 'remote',
            'remote_host_ip': '127.0.0.1',
            'remote_host_port': '8080',
        })
        response = heatmap_stream(request)
        self.assertIsInstance(response, StreamingHttpResponse)

    def test_opencv_not_installed(self):
        """Even without OpenCV, the response is a StreamingHttpResponse."""
        import builtins
        original_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == 'cv2':
                raise ImportError("cv2 not found")
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            request = self.factory.get('/stream/heatmap/')
            response = heatmap_stream(request)
            self.assertIsInstance(response, StreamingHttpResponse)
