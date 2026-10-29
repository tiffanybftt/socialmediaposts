# AI Summit Group: "Using Claude With Confidence"

A 3:20 team training video on privacy, security and smart habits with Claude. Built from code, so every
slide, caption and the voice-over come from the files in this folder and can be changed and re-rendered.

| File | What it is |
|---|---|
| `ai-summit-claude-training.mp4` | The finished video (1080p, narration + soft music + burned-in captions) |
| `RECORDING_SCRIPT.pdf` / `.md` | A script to record in your own voice: 11 short takes with delivery notes |
| `SCRIPT.md` | The narration, scene by scene (generated from `episode/script.json`) |
| `voice-samples/` | Sample clips of the same line in different AI voices |
| `build/captions.srt` | Subtitles as a separate file, for platforms that take them |
| `episode/script.json` | Source of truth for the words, scene order and pacing |
| `video.html` | All visuals. Brand colors and fonts are the variables at the top (`:root`) |
| `assets/` | Your logo (original + cut-out layers) and the Montserrat font files |
| `tools/` | The build scripts |

## Before this goes to the team

1. **Watch and listen all the way through.** The narration is an AI voice and was not heard by the author, so
   check pronunciation, pacing and that every caption matches what is said.
2. **Confirm which Claude plan the team is on.** The "Your plan" slide and one narration line say that on
   Team and Enterprise plans Anthropic does not train on conversations by default, and that personal accounts
   have a Model Improvement setting. If the team is on a different plan, change `p2` in `episode/script.json`
   and the "Your plan" card in `video.html`.
3. **Confirm "approved work account."** The video tells people to stick to approved tools and accounts. Make sure
   that matches what the company has actually approved.
4. If this turns into a written company policy, have a lawyer in your state review it. This video is training,
   not legal advice.

## Where the facts came from

| Claim in the video | Source |
|---|---|
| Personal (Free/Pro/Max) chats are used for training only if the Model Improvement setting is on; Incognito chats are excluded | [Anthropic privacy center: Is my data used for model training?](https://privacy.claude.com/en/articles/10023580-is-my-data-used-for-model-training) |
| Connected apps: Claude inherits each person's permissions; owners control what is enabled | [Anthropic help center: connectors](https://support.claude.com/en/articles/11176164-use-connectors-to-extend-claude-s-capabilities) |
| Team/Enterprise: conversations, files and connected data are not used for training by default | [Claude Academy: Enterprise administrator guide](https://academy.claude.com/tutorials/claude-enterprise-administrator-guide), seen as a search summary. Verify against your plan's Commercial Terms |

The traffic-light categories (never share passwords, API keys, card/bank numbers, government IDs; ask first for
client, confidential, HR/payroll/medical information) follow the company AI-use policy. The team's own survey
answers are summarized as four themes without names.

## Rebuilding

Needs `ffmpeg`, Node + Playwright, and Python with `kokoro-onnx` and `numpy`. The voice is set in
`episode/script.json` (`"engine": "kokoro"`, `"voice": "af_heart"`, `"speed": 0.95`). Pick any voice name from
`voice-samples/`. The older Piper engine still works with `"engine": "piper"`, `"voice": "en_US-lessac-high"`
and `"length_scale": 1.12`.

```bash
python tools/prep_logo.py                       # logo cut-outs (only if the logo changes)
python tools/build_audio.py /path/to/voice-models  # narration + build/timeline.js + captions
python tools/make_music.py 201 calm build/timeline.json   # music bed, same length as the narration (see build/timeline.json)
node tools/render.js 30 4                        # silent video, 4 parallel browsers (~5 min)
tools/mix.sh                                     # mix audio onto the video
```

Notes:

- Re-run `build_audio.py` and `render.js` together. The voice varies slightly run to run, so the timeline changes
  and the visuals must be re-rendered to match.
- `node tools/snap.js out_dir 12 40 95` saves still frames at those seconds to check layout fast.
- **Using a human voice.** Record the 11 takes in `RECORDING_SCRIPT.pdf` (Take_1 ... Take_7, with 6a to 6e). Then,
  instead of `build_audio.py`:

  ```bash
  python tools/transcribe_takes.py /path/to/takes /path/to/rec   # local speech-to-text with word timings
  python tools/align_takes.py /path/to/rec                      # finds each script sentence; reports any changes
  python tools/build_audio_human.py /path/to/rec                # level-matches takes, writes the same build files
  ```

  Then run `make_music.py`, `render.js` and `mix.sh` as above. `episode/takes.json` says which sentences belong to
  which take. If you ad-lib, edit that sentence in `episode/script.json` to match what you said so the captions agree.
  The closing card says "AI-generated voice" only when the voice is AI.
- **Calm or upbeat.** Add `--style upbeat` to `build_audio_human.py` (it trims long pauses and switches the video to
  snappier animations plus a confetti burst on the closing logo), and run `python tools/make_music.py <seconds> upbeat
  build/timeline.json` for a 108 BPM groove whose whooshes, riser and crash line up with the scene changes and the logo
  reveal. Mix it with `MUSIC_VOL=0.24 DUCK_RATIO=3 tools/mix.sh out.mp4` so the music is a little more present.
  The first take of each scene is snapped to where your voice really starts, because speech-to-text places it early.
- **Record quietly.** Keep the input level low enough that the meter never hits red. Clipped takes can't be repaired.
  Do a 10-second test and listen back before recording the full thing.

## Turning this into a reusable base

For the next episode, keep `video.html`'s brand variables, the chrome (logo bar, caption pill, progress trail), the
intro/close lockups and the card/row/panel styles. Replace the scene sections and `episode/script.json`. The
progress trail's labels (Welcome, Learn, Answers, Summit) can stay as the series structure. The trail sizes itself
to however many steps are in the page, so a "Discuss" step can be added back later by adding a scene and a label.
