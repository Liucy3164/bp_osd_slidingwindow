# BP-OSD with slidingwindow-

[1]: https://github.com/quantumgizmos/bp_osd
[2]: https://journals.aps.org/pra/abstract/10.1103/PhysRevA.110.012453


BP-OSD decoding with a sliding window for QLDPC codes, modified from the open-source implementation in [[1]]. I am trying to reproduce the results in [[2]] by adding a sliding window and measurement errors.

I implemented the sliding-window code and provide an example for the [[625, 25, 8]] LDPC code, showing lifetime versus error rate.

To run it, replace css_decode_sim.py in [[1]] with my modified version, and then run my new sliding-window example script to evaluate the performance.
