"""SaladCloud control-plane helpers.

Everything here is optional. Without SALAD_API_KEY (or without the SDK installed) every
function no-ops and returns False, so main.py behaves exactly as it does off SaladCloud.

Two variables are injected into every container by the platform:

    SALAD_MACHINE_ID           the node this replica is running on
    SALAD_CONTAINER_GROUP_ID   the group it belongs to

They identify the container but are not enough to call the API, whose paths are keyed by
organization / project / container-group *name*. Those three have to be supplied as env vars
on the container group itself, alongside the API key.

What the API can and cannot do, which shapes what is worth building on it:

  stop_container_group      stops every replica. The right call once the dataset is complete.
  set_replicas              scales the group. Salad chooses which replica dies, not you.
  reallocate_self           moves this replica to a DIFFERENT node. There is no way to request
                            a specific node, so it cannot be used to return to a card whose
                            checkpoint you hold -- it is a lottery over the whole node pool.
  restart / recreate        put the same node back to work rather than releasing it.

Notably absent: any way to terminate a single instance while the others keep running.
"""

import os

MACHINE_ID = os.environ.get("SALAD_MACHINE_ID")
CONTAINER_GROUP_ID = os.environ.get("SALAD_CONTAINER_GROUP_ID")

API_KEY = os.environ.get("SALAD_API_KEY")
ORG = os.environ.get("SALAD_ORG")
PROJECT = os.environ.get("SALAD_PROJECT")
GROUP = os.environ.get("SALAD_CONTAINER_GROUP_NAME")


def configured() -> bool:
    """True when every field the REST paths need is present."""
    return all((API_KEY, ORG, PROJECT, GROUP))


def on_salad() -> bool:
    """True when running inside a SaladCloud container, whether or not the API is configured."""
    return MACHINE_ID is not None


def _service():
    """The SDK's container-groups service, or None if unavailable. Imported lazily so the
    profiler runs anywhere without the dependency."""
    if not configured():
        return None
    try:
        from salad_cloud_sdk import SaladCloudSdk
        return SaladCloudSdk(api_key=API_KEY, timeout=30000).container_groups
    except Exception as e:
        print(f"[salad] SDK unavailable, control-plane calls disabled: {type(e).__name__}: {e}")
        return None


def _call(what: str, fn) -> bool:
    """Run one control-plane call, reporting rather than raising. A failure here must never
    take down a run that has already produced data."""
    svc = _service()
    if svc is None:
        return False
    try:
        fn(svc)
        print(f"[salad] {what}", flush=True)
        return True
    except Exception as e:
        print(f"[salad] {what} FAILED: {type(e).__name__}: {e}", flush=True)
        return False


def stop_container_group() -> bool:
    """Stop every replica in the group. Call when there is no work left for anyone -- otherwise
    Salad restarts each finished container forever."""
    return _call("stopped container group", lambda s: s.stop_container_group(
        organization_name=ORG, project_name=PROJECT, container_group_name=GROUP))


def set_replicas(n: int) -> bool:
    """Scale the group. Which replica Salad terminates is its choice, not ours, so this cannot
    be used to retire the specific node that just finished."""
    from salad_cloud_sdk.models import ContainerGroupPatch
    return _call(f"set replicas to {n}", lambda s: s.update_container_group(
        organization_name=ORG, project_name=PROJECT, container_group_name=GROUP,
        request_body=ContainerGroupPatch(replicas=n)))


def reallocate_self() -> bool:
    """Move this replica to a different node. Deliberately not used for checkpoint recovery:
    the destination is chosen by Salad, so this cannot bring us back to a card we hold a
    checkpoint for."""
    if not MACHINE_ID:
        return False
    return _call(f"reallocated away from {MACHINE_ID}", lambda s: s.reallocate_container_group_instance(
        organization_name=ORG, project_name=PROJECT, container_group_name=GROUP,
        container_group_instance_id=MACHINE_ID))


def instance_count() -> int | None:
    """How many replicas are currently running, or None if it cannot be determined."""
    svc = _service()
    if svc is None:
        return None
    try:
        r = svc.list_container_group_instances(
            organization_name=ORG, project_name=PROJECT, container_group_name=GROUP)
        return len(getattr(r, "instances", []) or [])
    except Exception as e:
        print(f"[salad] instance listing failed: {type(e).__name__}: {e}")
        return None
