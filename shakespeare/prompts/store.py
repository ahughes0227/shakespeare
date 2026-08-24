"""Load and verify pinned prompt artifacts."""

from __future__ import annotations

from pathlib import Path

import yaml

from ..contracts import PromptArtifact, content_digest

PROMPT_ROOT = Path(__file__).resolve().parents[2] / "_prompts"


class PromptStoreError(RuntimeError):
    pass


class PromptStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or PROMPT_ROOT

    def path_for(self, signature_id: str, version: str) -> Path:
        return self.root.joinpath(*signature_id.split(".")) / f"{version}.yaml"

    def load(self, signature_id: str, version: str) -> PromptArtifact:
        """Load a pinned artifact and verify it against its recorded digest.

        Verification matters because a prompt version is part of the workflow digest: a
        prompt edited in place without a version bump would silently change behaviour
        while claiming to be the same run.
        """
        path = self.path_for(signature_id, version)
        if not path.is_file():
            raise PromptStoreError(f"no prompt artifact for {signature_id}@{version} at {path}")

        payload = yaml.safe_load(path.read_text()) or {}
        declared = payload.pop("digest", None)
        artifact = PromptArtifact.model_validate(
            {**payload, "signature_id": signature_id, "version": version}
        )
        if declared is not None and declared != artifact.digest():
            raise PromptStoreError(
                f"{signature_id}@{version} does not match its recorded digest; "
                f"bump the version rather than editing an artifact in place"
            )
        return artifact

    def save(self, artifact: PromptArtifact) -> Path:
        path = self.path_for(artifact.signature_id, artifact.version)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = artifact.model_dump(mode="json", exclude={"signature_id", "version"})
        payload["digest"] = artifact.digest()
        path.write_text(yaml.safe_dump(payload, sort_keys=True, allow_unicode=True))
        return path

    def versions(self, signature_id: str) -> tuple[str, ...]:
        directory = self.root.joinpath(*signature_id.split("."))
        if not directory.is_dir():
            return ()
        return tuple(sorted(item.stem for item in directory.glob("*.yaml")))


def digest_of_artifact(artifact: PromptArtifact) -> str:
    return content_digest(artifact)
