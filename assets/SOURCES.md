# Visual sources

All presentation assets are stored inside this repository. No external badge, tracking image, font service or image host is used. Local storage does not mean that every image or its underlying research was originally created by Sky.

## Shared materials and permission

Sky confirms that **Zhenyu Wu authorized inclusion of the team data, reference content and images he supplied in the original public GitHub research portfolio, whose materials and credits are preserved in this private copy**. Zhenyu is credited as the provider; original authorship remains with the team and upstream creators, including Huaze Liu where documented. Existing licenses and notices remain applicable. See [access and permission](../CONFIDENTIALITY.md) and the [source and permission record](../docs/AI_AND_SOURCE_DISCLOSURE.md#material-permissions-and-provenance).

## Portfolio figures and recordings

The assets below are presentation derivatives or recordings from the preserved AI-assisted evaluation. Their plots and layout were produced with Codex assistance; they do not establish sole authorship of the underlying controller, navigation method, shared data or reference visuals. ChatGPT helped integrate and draft the accompanying explanations.

These derivatives are distinguished from borrowed team figures and images, which retain their source credits.

- `research-cover.svg`: simplified display of simulator-truth positions from the staggered-baffle seed-0 canonical evaluation record; map boundaries and dividers come from its authored manifest. Points are sampled for display; the raw record is unchanged.
- `pilot-at-a-glance.svg`: counts from the four completed canonical evaluation records. The interrupted attempt is excluded and remains preserved separately.
- `arrival-confidence.svg` / `.png`: final runtime particle-filter arrival probabilities from `projects/tactile-navigation/outputs/sky_results.json`; the common 95% criterion is shown on a zero-based axis.
- `section-*.svg`: typography and layout only.
- `offset-opening-trace.png` / `.svg`: every recorded planar localization-error sample and probe-event time from the completed offset-opening seed-0 evaluation. The separate contact/miss rows use recorded outcomes; temporal association is not a causal effect estimate. Source hash, counts, and independently recomputed RMSE are recorded in `presentation/navigation_trace_audit.json`.
- Robot thumbnails in the README and video gallery are existing frames/contact sheets from the executed simulations, linked without alteration.

Rebuild with Python and Matplotlib: `python presentation/build_assets.py`.

Rebuild the episode figure with Python and Matplotlib: `python presentation/build_navigation_trace.py`. This reads preserved data and does not launch simulation or training.
