from pathlib import Path

import pytest

from aether_rl.configs.server import S3PolicyDistributionConfig
from aether_rl.coordinator.policy_distribution import PolicyDistributionError, S3PolicyDistributor
from aether_rl.protocol import policy_manifest_digest
from tests.unit.worker.test_policy_runtime import published_policy


class MissingObject(Exception):
    response = {"ResponseMetadata": {"HTTPStatusCode": 404}, "Error": {"Code": "NoSuchKey"}}


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.puts = 0

    def head_object(self, *, Bucket, Key):
        try:
            return self.objects[(Bucket, Key)]
        except KeyError:
            raise MissingObject from None

    def put_object(self, *, Bucket, Key, Body, ContentLength, ContentType, CacheControl, Metadata):
        data = Body.read() if hasattr(Body, "read") else Body
        assert len(data) == ContentLength
        self.puts += 1
        self.objects[(Bucket, Key)] = {
            "ContentLength": ContentLength,
            "ContentType": ContentType,
            "CacheControl": CacheControl,
            "Metadata": Metadata,
            "Body": data,
        }

    def generate_presigned_url(self, operation, *, Params, ExpiresIn):
        assert operation == "get_object"
        return f"https://cdn.test/{Params['Bucket']}/{Params['Key']}?expires={ExpiresIn}"


def test_s3_policy_distribution_is_verified_idempotent_and_presigned(tmp_path: Path):
    manifest, policy_dir = published_policy(tmp_path)
    client = FakeS3()
    config = S3PolicyDistributionConfig(bucket="policies", endpoint_url="https://r2.test", presign_ttl_seconds=60)
    distributor = S3PolicyDistributor(config, client=client, clock=lambda: 10)

    distributor.publish(manifest, policy_dir)
    assert client.puts == 4
    distributor.publish(manifest, policy_dir)
    assert client.puts == 4

    locations = distributor.locations(manifest)
    assert locations.policy_id == manifest.policy_id
    assert locations.policy_digest == policy_manifest_digest(manifest)
    assert locations.expires_at == 70
    assert [file.name for file in locations.files] == ["adapter_config.json", "adapter_model.safetensors"]

    object_key = next(key for key in client.objects if key[1].endswith("adapter_model.safetensors"))
    client.objects[object_key]["ContentLength"] += 1
    with pytest.raises(PolicyDistributionError, match="conflicts"):
        distributor.publish(manifest, policy_dir)
