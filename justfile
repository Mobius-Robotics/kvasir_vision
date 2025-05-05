export OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS := "0"

run command:
    uv run python -m kvasir_vision "{{command}}"
