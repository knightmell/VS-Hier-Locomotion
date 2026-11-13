The root MIT license applies to original software in this release.

- `vendor/stable-baselines3/` contains a locomotion dependency subset from
  [the DSRL SB3 fork](https://github.com/ajwagen/stable-baselines3-dsrl) and
  [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), together with
  VS-Hier extensions. The upstream MIT `LICENSE` and `NOTICE` are retained.
  Package exports and packaging metadata are reduced to the released dependency
  subset; the three-critic core, replay, and schedule retain their source behavior.
- `vendor/dppo/` contains the frozen MLP/DDIM decoder dependency subset from
  [the DSRL DPPO fork](https://github.com/ajwagen/dppo-dsrl), based on
  [DPPO](https://github.com/irom-princeton/dppo). Its MIT license is retained.
  Environment integrations and fine-tuning agents are not bundled.
- The unlicensed DSRL root integration files are not redistributed. The released
  `loco_env.py` and `policy_loader.py` provide standalone locomotion interfaces.
- Pretrained weights and datasets are not redistributed. Obtain them from their
  original publishers and observe the applicable asset terms.
- Paper figures and demonstration previews in `assets/` are presentation assets
  provided by the authors, with copyright retained; they are excluded from the
  software MIT license. Their display does not release the associated private
  robot implementation, source photographs, or training data.
