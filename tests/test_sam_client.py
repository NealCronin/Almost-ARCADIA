from unittest.mock import Mock, patch

import numpy as np
import pytest
import requests

from core.errors import InferenceError
from core.inference.sam_client import SAMClient
from core.services.specs import ServiceEndpoint


@patch("core.inference.sam_client.requests.post")
def test_segment_parses_response(mock_post: Mock) -> None:
    response = Mock()
    response.json.return_value = {
        "masks": [[[1]]],
        "labels": ["object"],
        "confidences": [0.9],
        "bounding_boxes": [[0, 0, 1, 1]],
    }
    mock_post.return_value = response

    result = SAMClient(ServiceEndpoint("127.0.0.1", 8090, "sam3")).segment(
        np.zeros((4, 4, 3), dtype=np.uint8),
        ["object"],
    )

    assert result.labels == ["object"]
    assert result.confidences == [0.9]


@patch("core.inference.sam_client.requests.post")
def test_segment_marks_request_failure_as_sam(mock_post: Mock) -> None:
    mock_post.side_effect = requests.ConnectionError("unreachable")
    client = SAMClient(ServiceEndpoint("127.0.0.1", 8090, "sam3"))

    with pytest.raises(InferenceError) as exc_info:
        client.segment(np.zeros((4, 4, 3), dtype=np.uint8), ["object"])

    assert exc_info.value.service_type == "sam3"
