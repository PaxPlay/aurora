# Rendering framework for toy renderers and visualization stuff

Right now, this is kind of a toy project with a bunch of technologies I have not worked with before. A live demo of the
pathtracer can be found on [github pages](https://paxplay.github.io/aurora). Since WebGPU is now enabled in both
Chromium and Firefox, you should be able to test it out online.

To run the pathtracer example use `cargo run --example pathtracer`. Alternatively you can use
[trunk](https://trunkrs.dev/) to compile against the wasm/WebGPU backend using `trunk serve --example pathtracer`.
