"""List audio input/output devices. Usage: uv run -m character_eng.devices"""

from character_eng.voice import list_audio_devices


def main():
    for d in list_audio_devices():
        io = ("IN " if d["max_input_channels"] > 0 else "   ") + (
            "OUT" if d["max_output_channels"] > 0 else "   "
        )
        print(f"{d['index']:3d}  {io}  {d['name']}")


if __name__ == "__main__":
    main()
