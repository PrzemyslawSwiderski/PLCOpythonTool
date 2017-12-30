SELECT
  age,
  psa_level0,
  pros_gleason,
  dth_cat
FROM
  prostate_screening.prostate_overall
WHERE
  is_dead = 1.0 AND f_cancersite = 1.0 AND dth_cat = 1.0
  AND (psa_level0 != 0.0) AND (psa_level0 <= 35.0) AND (pros_gleason NOT IN (0.0, 99.0))