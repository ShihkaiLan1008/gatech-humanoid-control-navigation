# Visual sources

All presentation assets are stored inside this private repository. No third-party badge, tracking image, font service or image host is used.

- `research-cover.svg`: simplified display of simulator-truth positions from the staggered-baffle seed-0 canonical evaluation record; map boundaries and dividers come from its authored manifest. Points are sampled for display; the raw record is unchanged.
- `pilot-at-a-glance.svg`: counts from the four completed canonical evaluation records. The interrupted attempt is excluded and remains preserved separately.
- `arrival-confidence.svg` / `.png`: final runtime particle-filter arrival probabilities from `projects/tactile-navigation/outputs/sky_results.json`; the common 95% criterion is shown on a zero-based axis.
- `section-*.svg`: typography and layout only.
- `offset-opening-trace.png` / `.svg`: every recorded planar localization-error sample and probe-event time from the completed offset-opening seed-0 evaluation. The separate contact/miss rows use recorded outcomes; temporal association is not a causal effect estimate. Source hash, counts, and independently recomputed RMSE are recorded in `presentation/navigation_trace_audit.json`.
- Robot thumbnails in the README and video gallery are existing frames/contact sheets from the executed simulations, linked without alteration.

Rebuild with Python and Matplotlib: `python presentation/build_assets.py`.

Rebuild the episode figure with Python and Matplotlib: `python presentation/build_navigation_trace.py`. This reads preserved data and does not launch simulation or training.
