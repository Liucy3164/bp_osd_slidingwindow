# BP-OSD with sliding window

[1]: https://github.com/quantumgizmos/bp_osd
[2]: https://journals.aps.org/pra/abstract/10.1103/PhysRevA.110.012453


BP-OSD decoding with a sliding window for QLDPC codes, modified from the open-source implementation in [[1]]. I am trying to reproduce the results in [[2]] by adding a sliding window and measurement errors.

Example for the [[625, 25, 8]] LDPC code, showing lifetime versus error rate.

To run it, replace css_decode_sim.py in [[1]] with my modified version, and then run qldpc_decode_[[625,25,8]].ipynb as new example script to evaluate the performance. One could add more data points for further better results.
