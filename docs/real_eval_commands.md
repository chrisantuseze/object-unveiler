# Real-robot evaluation: commands

The method is chosen on the lab PC: start the server for one method, run the robot client, stop the server, start
the next. The client command is the same for every method; only its session name changes.

Scenes and sheets: `twin/real_eval_layouts/` (`layouts.pdf` for Task 1, `layouts_task2.pdf` for Task 2).

## Lab PC: one server per method

From the repo root, in the `unveiler` env.

```bash
python -m robot.server --jetson-ip 192.168.0.8 --warp 88 71 501 38 639 479 54 479 --method ours            # SRE, expert iteration in the twin (AlphaZero style)
python -m robot.server --jetson-ip 192.168.0.8 --warp 88 71 501 38 639 479 54 479 --method il              # SRE, imitation only
python -m robot.server --jetson-ip 192.168.0.8 --warp 88 71 501 38 639 479 54 479 --method gpt4o           # needs OPENAI_API_KEY
python -m robot.server --jetson-ip 192.168.0.8 --warp 88 71 501 38 639 479 54 479 --method clip
python -m robot.server --jetson-ip 192.168.0.8 --warp 88 71 501 38 639 479 54 479 --method heuristic
python -m robot.server --jetson-ip 192.168.0.8 --warp 88 71 501 38 639 479 54 479 --method always_target   # grasps the target at once
```

Also available: `--method ppo` (the PPO SRE) and `--method random`.

## Jetson: the client

From the `dofbot-controller` root, with the robot stack and rosbridge running. Copy the scene files once:
`scp twin/real_eval_layouts/scenes*.yaml <user>@192.168.0.8:<dofbot-controller>/unveiler/`

```bash
# Task 1: one target (20 scenes)
python3 unveiler/unveiler_session.py --jetson_ip 127.0.0.1 --executor robot \
    --scenes unveiler/scenes.yaml --session_name task1_<method>

# Task 2: two targets in order (10 scenes)
python3 unveiler/unveiler_session.py --jetson_ip 127.0.0.1 --executor robot \
    --scenes unveiler/scenes_task2.yaml --session_name task2_<method>
```

`<method>` is the name given to the running server: `task1_ours`, `task1_il`, `task1_gpt4o`, `task1_clip`,
`task1_heuristic`, `task1_always_target`. Never reuse a session name: the server writes to
`save/real_eval/<session_name>/` and would overwrite it.

Until the client is changed (below) it still wants a method list. Add `--methods sre_il`; the server ignores the
name and runs its own method.

Dry run without arm motion: the same command with `--executor none`.

## Jetson client changes

1. **No methods in the client.** Remove `--methods`, the loop over methods and the client-side `always_target`
   branch. Each scene runs once per session. Record the method from the server: every `select` reply carries
   `method` (`ours`, `il`, ...), and so does `ping()`.
2. **A grasp at the target is final.** Once the target is picked, the episode ends, whether the grasp worked or
   not: success if the block is in the bin, failure otherwise. No second attempt. In Task 2 a failed grasp at
   either target ends the episode.

Nothing else changes: the wire format is the same, and `always_target` is now answered by the server like any
other method (when the target is hidden it returns no selection, which the client already logs as a failure).

## Checks on the lab PC

```bash
python -m twin.make_eval_layouts --check grid --frame <frame.jpg>                    # table marks, empty table
python -m twin.make_eval_layouts --check C1-03 --frame save/real_eval/<session>     # a built scene vs its sheet
```

## Filming for the video

Four layouts: C2-02 and F-02 (Task 1), T2a-01 and T2b-01 (Task 2). Copy `scenes_video.yaml` and
`scenes_video_task2.yaml` to the Jetson as above. These sessions are for the video only and do not enter the results.

```bash
# Lab PC, one method at a time: --method ours, then heuristic, then gpt4o (same server command as above)

# Jetson
python3 unveiler/unveiler_session.py --jetson_ip 127.0.0.1 --executor robot \
    --scenes unveiler/scenes_video.yaml --session_name video_task1_<method>
python3 unveiler/unveiler_session.py --jetson_ip 127.0.0.1 --executor robot \
    --scenes unveiler/scenes_video_task2.yaml --session_name video_task2_<method>
```

Film `ours` on both files, `heuristic` on both, and `gpt4o` on Task 2 only.
