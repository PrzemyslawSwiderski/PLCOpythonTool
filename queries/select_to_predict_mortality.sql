SELECT
  dx_psa,
  pros_gleason,
  dth_days
FROM
  prostate_screening.prostate_overall
WHERE dx_psa > 0.0 AND dx_psa < 40.0 AND is_dead = 1.0 AND (pros_gleason NOT IN (0.0, 99.0))