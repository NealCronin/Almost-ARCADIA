from unittest.mock import Mock, patch

import numpy as np

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

    client = SAMClient(
        ServiceEndpoint(
            host="127.0.0.1",
            port=8090,
            service_type="sam3",
        )
    )

    result = client.segment(
        np.zeros((4, 4, 3), dtype=np.uint8),
        ["object"],
    )

    assert result.labels == ["object"]
    assert result.confidences == [0.9]
