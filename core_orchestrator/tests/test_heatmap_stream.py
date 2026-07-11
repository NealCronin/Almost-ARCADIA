"""
Tests for heatmap_stream endpoint.
"""
import os
import unittest
from unittest.mock import patch, MagicMock

from django.test import TestCase, RequestFactory
from django.http import StreamingHttpResponse

from core_orchestrator.views import heatmap_stream


class HeatmapStreamTests(TestCase):
    """Test cases for heatmap_stream view."""

    def setUp(self):
        """Set up test fixtures."""
        os.environ.setdefault('ALMOST_ARCADIA_CONFIG_DIR', 'C:/Users/Neal/AppData/Local/Temp/test_arcadia')
        self.factory = RequestFactory()

    def test_heatmap_stream_returns_streaming_response(self):
        """Test that heatmap_stream returns a StreamingHttpResponse."""
        request = self.factory.get('/stream/heatmap/')
        response = heatmap_stream(request)
        
        self.assertIsInstance(response, StreamingHttpResponse)
        self.assertEqual(response['Content-Type'], 'multipart/x-mixed-replace; boundary=frame')

    def test_heatmap_stream_with_local_mode(self):
        """Test heatmap_stream with local routing mode."""
        request = self.factory.get('/stream/heatmap/', {
            'routing_mode': 'local',
            'dataset_path': '',
        })
        response = heatmap_stream(request)
        
        self.assertIsInstance(response, StreamingHttpResponse)

    def test_heatmap_stream_with_remote_mode(self):
        """Test heatmap_stream with remote routing mode."""
        request = self.factory.get('/stream/heatmap/', {
            'routing_mode': 'remote',
            'remote_host_ip': '127.0.0.1',
            'remote_host_port': '8080',
        })
        response = heatmap_stream(request)
        
        self.assertIsInstance(response, StreamingHttpResponse)

    def test_heatmap_stream_with_all_params(self):
        """Test heatmap_stream with all query parameters."""
        request = self.factory.get('/stream/heatmap/', {
            'dataset_path': '/path/to/video.mp4',
            'routing_mode': 'local',
            'llm_model_path': '/path/to/model.gguf',
            'sam3_weights_path': '/path/to/sam3.pt',
            'remote_host_ip': '192.168.1.100',
            'remote_host_port': '8080',
        })
        response = heatmap_stream(request)
        
        self.assertIsInstance(response, StreamingHttpResponse)

    def test_heatmap_stream_opencv_not_installed(self):
        """Test heatmap_stream handles missing OpenCV gracefully."""
        # Only mock the cv2 module import, not __import__ globally
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


if __name__ == '__main__':
    unittest.main()