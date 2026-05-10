#!/usr/bin/env python3
"""追跡の途中で周期的に text prompt を再実行して補正する benchmark。"""

from __future__ import annotations

import argparse

from video_text_prompt_benchmark_utils import (
    Timer,
    add_common_video_args,
    build_video_model,
    finalize_summary,
    get_end_frame,
    init_video_state,
    make_frame_record,
    print_summary,
    sync_cuda,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark SAM3 tracking with periodic text-prompt refresh"
    )
    add_common_video_args(parser, include_refresh=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.refresh_interval <= 0:
        raise ValueError("refresh interval must be positive")

    model = build_video_model(args)
    state = init_video_state(model, args)
    end_frame = get_end_frame(state, args)

    frame_records = []
    refresh_count = 0
    current_frame = args.start_frame

    with Timer() as total_timer:
        while current_frame <= end_frame:
            segment_end = min(current_frame + args.refresh_interval - 1, end_frame)

            with Timer() as refresh_timer:
                prompt_frame_idx, prompt_output = model.add_prompt(
                    state,
                    frame_idx=current_frame,
                    text_str=args.prompt,
                )
                sync_cuda(args.device)
            frame_records.append(
                make_frame_record(
                    frame_idx=prompt_frame_idx,
                    elapsed_sec=refresh_timer.elapsed,
                    output=prompt_output,
                    phase="refresh_detect",
                    refresh_index=refresh_count,
                )
            )

            track_span = segment_end - current_frame
            if track_span > 0:
                for frame_idx, output in model.propagate_in_video(
                    state,
                    start_frame_idx=current_frame + 1,
                    max_frame_num_to_track=track_span - 1,
                    reverse=False,
                ):
                    sync_cuda(args.device)
                    frame_records.append(
                        make_frame_record(
                            frame_idx=frame_idx,
                            elapsed_sec=0.0,
                            output=output,
                            phase="track",
                            refresh_index=refresh_count,
                        )
                    )

            refresh_count += 1
            current_frame = segment_end + 1

    summary = finalize_summary(
        mode="track_with_periodic_text_refresh",
        prompt=args.prompt,
        input_path=args.input,
        frame_records=frame_records,
        total_elapsed_sec=total_timer.elapsed,
        extra={
            "refresh_count": refresh_count,
            "refresh_interval": args.refresh_interval,
        },
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
