import time

from app.ui.camera_monitor import (
    CameraMonitor,
    CameraState,
)


def main():

    print(
        "\nMemorAi Camera Monitor Test\n"
    )

    monitor = CameraMonitor(
        camera_index=0,
        target_fps=10,
    )

    # --------------------------------------------------
    # START
    # --------------------------------------------------

    print(
        "1. Starting camera..."
    )

    monitor.start()

    time.sleep(
        3
    )

    print(
        "State:",
        monitor.state.value,
    )

    assert monitor.state in {
        CameraState.RUNNING,
        CameraState.STARTING,
    }

    # --------------------------------------------------
    # FRAME
    # --------------------------------------------------

    frame = (
        monitor.get_latest_frame()
    )

    assert frame is not None

    print(
        "PASS: Camera frame received."
    )

    # --------------------------------------------------
    # PAUSE
    # --------------------------------------------------

    print(
        "\n2. Pausing..."
    )

    monitor.pause()

    time.sleep(
        1
    )

    assert (
        monitor.state
        == CameraState.PAUSED
    )

    print(
        "PASS: Camera paused."
    )

    # --------------------------------------------------
    # RESUME
    # --------------------------------------------------

    print(
        "\n3. Resuming..."
    )

    monitor.resume()

    time.sleep(
        2
    )

    assert (
        monitor.state
        == CameraState.RUNNING
    )

    print(
        "PASS: Camera resumed."
    )

    # --------------------------------------------------
    # STOP
    # --------------------------------------------------

    print(
        "\n4. Stopping..."
    )

    monitor.stop()

    assert (
        monitor.state
        == CameraState.STOPPED
    )

    print(
        "PASS: Camera stopped."
    )

    # --------------------------------------------------
    # START AGAIN
    # --------------------------------------------------

    print(
        "\n5. Starting again..."
    )

    monitor.start()

    time.sleep(
        2
    )

    assert (
        monitor.get_latest_frame()
        is not None
    )

    print(
        "PASS: Camera restarted."
    )

    monitor.stop()

    print(
        "\nAll CameraMonitor tests passed."
    )


if __name__ == "__main__":
    main()