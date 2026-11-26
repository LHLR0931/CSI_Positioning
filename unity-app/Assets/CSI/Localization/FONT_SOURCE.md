# Bundled UI font

`Resources/Localization/NotoSansSC-Regular.otf` is the unmodified Simplified
Chinese subset of Noto Sans CJK, version 2.004, downloaded on 2026-10-26.
It contains Latin and Simplified Chinese glyphs and is used in both UI languages.

- Official repository: https://github.com/notofonts/noto-cjk
- Original font: https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/SubsetOTF/SC/NotoSansSC-Regular.otf
- Official license: https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/LICENSE
- License: SIL Open Font License 1.1; the complete unmodified license is included
  at `Resources/Localization/OFL.txt` and is bundled as a Resources text asset.
- Font length: 8,331,336 bytes.
- Font SHA-256: `faa6c9df652116dde789d351359f3d7e5d2285a2b2a1f04a2d7244df706d5ea9`.
- License SHA-256: `6a73f9541c2de74158c0e7cf6b0a58ef774f5a780bf191f2d7ec9cc53efe2bf2`.

The font was retrieved from the official repository rather than copied from a
Windows font installation. No font outlines, names or embedded metadata were
modified. Its cmap was checked against all Chinese characters in the translation
catalog and the direction arrows / checkmark used by navigation. Unity import,
layout and device rendering are checked separately by the build owner.

`CsiLocalization` changes display labels only. Model IDs, source declarations,
input field contents, predictions, coordinates, replay data and CSV output retain
their original values. English is the upgrade default. A language selection is
saved under `csi.language`; tests can use `SetLanguage(chinese, false)` without
writing the preference. Unknown external server messages retain their original
details instead of inventing an interpretation.
