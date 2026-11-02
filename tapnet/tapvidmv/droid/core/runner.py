# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Sharded batch episode runner and skip signaling."""

import os
import traceback


class SkipEpisode(Exception):  # pylint: disable=g-bad-exception-name
  """Raised when an episode fails quality or stream preconditions."""


def finished(root, marker):
  if not os.path.isdir(root):
    return set()
  return {
      e
      for e in os.listdir(root)
      if os.path.exists(os.path.join(root, e, marker))
  }


def select_episodes(episode_ids, config):
  episode_ids = sorted(episode_ids)
  if config.runner.limit > 0:
    episode_ids = episode_ids[: config.runner.limit]
  return episode_ids[config.runner.rank :: config.runner.world_size]


def run_episodes(episode_ids, process, done, stage):
  """Runs process(episode_id) over unfinished episodes and logs summary counts."""
  done = set(done)
  todo = [episode_id for episode_id in episode_ids if episode_id not in done]
  already_done = len(episode_ids) - len(todo)
  print(
      f"{len(todo)} episodes to process"
      + (f" ({already_done} already done)" if already_done else "")
  )

  succeeded, skipped, failed = [], [], []
  for idx, episode_id in enumerate(todo):
    print(f"\n[{idx + 1}/{len(todo)}] Episode: {episode_id}")
    try:
      process(episode_id)
      succeeded.append(episode_id)
    except SkipEpisode as reason:
      print(f"  SKIP {episode_id}: {reason}")
      skipped.append(episode_id)
    except Exception:  # pylint: disable=broad-exception-caught
      failed.append(episode_id)
      traceback.print_exc()

  print(
      f"\n{stage}: {len(succeeded)} succeeded, {len(skipped)} skipped,"
      f" {len(failed)} failed of {len(todo)}."
  )
  for episode_id in failed:
    print(f"  FAILED {episode_id}")
