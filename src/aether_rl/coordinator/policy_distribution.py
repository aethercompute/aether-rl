from __future__ import annotations

import hashlib
import time
from pathlib import Path

from aether_rl.configs.server import S3PolicyDistributionConfig
from aether_rl.protocol import (
    PolicyFileLocation,
    PolicyLocations,
    PolicyManifest,
    canonical_json_bytes,
    policy_manifest_digest,
    sha256_digest,
)
from aether_rl.trainer.policy import POLICY_MANIFEST_NAME, verify_lora_policy


class PolicyDistributionError(RuntimeError):
    pass


class S3PolicyDistributor:
    def __init__(self, config: S3PolicyDistributionConfig, *, client=None, clock=time.time):
        self.config = config
        self.clock = clock
        if client is None:
            import boto3

            client = boto3.client(
                "s3",
                endpoint_url=None if config.endpoint_url is None else str(config.endpoint_url).rstrip("/"),
                region_name=config.region,
            )
        self.client = client

    def validate(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.config.bucket)
        except Exception as error:
            raise PolicyDistributionError(f"cannot access policy distribution bucket {self.config.bucket!r}") from error

    def publish(self, manifest: PolicyManifest, policy_dir: Path) -> None:
        if manifest.adapter is None:
            return
        verified = verify_lora_policy(policy_dir, expected=manifest)
        expected = {artifact.name: (artifact.size_bytes, artifact.digest) for artifact in verified.adapter.files}
        manifest_bytes = canonical_json_bytes(verified)
        expected[POLICY_MANIFEST_NAME] = (len(manifest_bytes), sha256_digest(manifest_bytes))

        for name, (size, digest) in expected.items():
            self._publish_file(
                self._key(verified, name),
                policy_dir / name,
                size=size,
                digest=digest,
                content_type="application/json" if name.endswith(".json") else "application/octet-stream",
            )

        commit = canonical_json_bytes(
            {
                "policy_id": verified.policy_id,
                "policy_digest": policy_manifest_digest(verified),
                "files": [
                    {"name": name, "size_bytes": size, "digest": digest} for name, (size, digest) in expected.items()
                ],
            }
        )
        self._publish_bytes(
            self._key(verified, "COMMITTED"),
            commit,
            digest=sha256_digest(commit),
            content_type="application/json",
        )

    def locations(self, manifest: PolicyManifest) -> PolicyLocations:
        if manifest.adapter is None:
            raise PolicyDistributionError("base policy has no downloadable adapter")
        expires_at = self.clock() + self.config.presign_ttl_seconds
        files = tuple(
            PolicyFileLocation(
                name=artifact.name,
                url=self.client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": self.config.bucket, "Key": self._key(manifest, artifact.name)},
                    ExpiresIn=self.config.presign_ttl_seconds,
                ),
            )
            for artifact in manifest.adapter.files
        )
        return PolicyLocations(
            policy_id=manifest.policy_id,
            policy_digest=policy_manifest_digest(manifest),
            expires_at=expires_at,
            files=files,
        )

    def _publish_file(self, key: str, path: Path, *, size: int, digest: str, content_type: str) -> None:
        if self._matches(key, size=size, digest=digest):
            return
        with open(path, "rb") as file:
            self.client.put_object(
                Bucket=self.config.bucket,
                Key=key,
                Body=file,
                ContentLength=size,
                ContentType=content_type,
                CacheControl="public, max-age=31536000, immutable",
                Metadata={"sha256": digest.removeprefix("sha256:")},
            )
        self._verify(key, size=size, digest=digest)

    def _publish_bytes(self, key: str, data: bytes, *, digest: str, content_type: str) -> None:
        if self._matches(key, size=len(data), digest=digest):
            return
        self.client.put_object(
            Bucket=self.config.bucket,
            Key=key,
            Body=data,
            ContentLength=len(data),
            ContentType=content_type,
            CacheControl="public, max-age=31536000, immutable",
            Metadata={"sha256": digest.removeprefix("sha256:")},
        )
        self._verify(key, size=len(data), digest=digest)

    def _matches(self, key: str, *, size: int, digest: str) -> bool:
        try:
            metadata = self.client.head_object(Bucket=self.config.bucket, Key=key)
        except Exception as error:
            response = getattr(error, "response", {})
            status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            code = str(response.get("Error", {}).get("Code", ""))
            if status == 404 or code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise
        actual = (
            metadata.get("ContentLength"),
            metadata.get("Metadata", {}).get("sha256"),
        )
        expected = (size, digest.removeprefix("sha256:"))
        if actual != expected:
            raise PolicyDistributionError(f"immutable policy object {key!r} conflicts with published content")
        self._verify_body(key, size=size, digest=digest)
        return True

    def _verify(self, key: str, *, size: int, digest: str) -> None:
        if not self._matches(key, size=size, digest=digest):
            raise PolicyDistributionError(f"policy object {key!r} was not durable after upload")

    def _verify_body(self, key: str, *, size: int, digest: str) -> None:
        response = self.client.get_object(Bucket=self.config.bucket, Key=key)
        body = response["Body"]
        hasher = hashlib.sha256()
        received = 0
        try:
            while chunk := body.read(1024 * 1024):
                received += len(chunk)
                hasher.update(chunk)
        finally:
            close = getattr(body, "close", None)
            if close is not None:
                close()
        if received != size or f"sha256:{hasher.hexdigest()}" != digest:
            raise PolicyDistributionError(f"policy object {key!r} bytes do not match published content")

    def _key(self, manifest: PolicyManifest, name: str) -> str:
        return f"{self.config.prefix}/runs/{manifest.run_id}/policies/{manifest.policy_id}/{name}"
