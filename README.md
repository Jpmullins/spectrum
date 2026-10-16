# Spectrum Viewer

A waterfall, power spectrum and ground-truth view over a synthetic band produced by
[`torchsig.spectrum`](../torchsig/torchsig/spectrum). It exists to answer one question by eye: *does the
synthetic stream look like what a receiver pointed at this band would actually deliver?*

![what it shows](docs/overview.png)

## Running it

The generator and the FFTs run where the GPU is; the browser can be anywhere.

```bash
# on spark01
source ~/.venvs/siggpt/bin/activate
python ~/code/rfml/spectrum/server.py --port 8080

# on your laptop
ssh -L 8080:127.0.0.1:8080 spark01
# then open http://127.0.0.1:8080
```

Nothing needs installing: it uses the standard library's HTTP server plus the numpy, torch and torchsig that
the siggpt environment already has. It binds to loopback only, so the SSH tunnel is the only way in.

Useful flags: `--band ism_2450|ism_915|uhf_lmr`, `--clutter urban_dense|urban|suburban|rural|open|indoor`,
`--tune-mhz`, `--decimation`, `--gain-db`, `--nfft`, `--row-ms`, `--seed`. Everything is also adjustable live
in the page.

## Reading the display

| | |
|---|---|
| **Waterfall** | Time runs downward; the newest row is at the bottom. One row is a whole number of FFT frames, so its duration is quantized and the bar reports what it realized. |
| **Power spectrum** | Blue is the current chunk, orange is peak hold. The dashed green line is the thermal floor, `kTB + NF`, in one bin. Signals should sit above it and the empty guard bands should sit on it. |
| **Truth overlay** | Every burst the generator placed, as a rectangle at its frequency, bandwidth and time. Above ~1200 bursts in view it switches from outlines to translucent fills, because a dense band really does fill in. The status panel says which mode is active. |
| **Channel plan** | Faint vertical lines at each technology's channel centres: Wi-Fi 1–11, the sixteen 802.15.4 channels, and Bluetooth LE's three advertising channels. |
| **Orange dashed line** | The LO/DC spur. The badge says how far it stands above the noise floor of one bin. |

Hovering reads out the frequency, the power in that bin, and any bursts under the cursor.

## Things worth looking for

These are the features that tell you the model is behaving, and roughly where to find them:

- **Wi-Fi crowds onto channels 1, 6 and 11.** Deployment preference, not a uniform draw over the eleven.
- **A data frame is followed by a very short acknowledgement** a SIFS later. That pairing is the texture of a
  real CSMA channel.
- **Beacons continue on channels with no traffic at all.** Most audible access points are idle; they are
  audible *because* they beacon every 102.4 ms.
- **Bluetooth LE advertises in three stripes** at 2402, 2426 and 2480 MHz, dodging Wi-Fi 1/6/11, and connected
  devices speckle the 37 data channels between them.
- **Bluetooth BR/EDR speckles the whole band** on a 625 µs slot clock, 1600 hops a second.
- **A microwave oven, if one is in the scene, keys at twice the mains frequency** — 8.3 ms on, 8.3 ms off at
  60 Hz — and its centre arcs up and back down within each burst as the magnetron's anode voltage follows the
  rectified half cycle. It is usually the loudest thing in the band.
- **Burst levels from one emitter vary.** That is Rician fading redrawn per burst, not noise.
- **The guard bands sit exactly on the thermal floor**, and the upper guard carries 3GPP band 53 above
  2483.5 MHz, because that allocation is real.

## Why it is slower than real time

Generating 100 MS/s of band costs far more than a radio does: the status bar reports the ratio, typically
0.06–0.15× real time. Stream time is what the waterfall shows, and the samples are exactly what a receiver
would have delivered — only the wall clock differs. Rendering is cached per
`(technology, kind, length, variant)`, so the ratio improves over the first few seconds and then holds.

## Layout

Drag the splitter beside the controls or above the spectrum to resize; double-click either to reset. `C` or
the button at the top right hides the controls. Sizes are remembered per browser. Rows are one device pixel
tall, so the history depth follows the window and the display's pixel ratio.

## What it is not

A viewer, not a receiver: there is no hardware in the loop. For what the model does and does not claim about
the band itself — which technologies are rendered faithfully and which are approximations — see the
`torchsig.spectrum` package docstrings and `catalog/additions.json`, where every added figure carries its
own basis and sources.
