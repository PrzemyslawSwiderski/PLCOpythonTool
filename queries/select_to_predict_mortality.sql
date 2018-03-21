SELECT
  dth_days,
  dx_psa,
  pros_gleason,
  pros_exitage,
  bmi_curc,
  weight_f,
  height_f,
--  dcf_unddeath,
--  pros_fh_age,
--  rectal_history,
--  surg_age,
  cig_years,
--  numbiopp,
--  asppd,
--  ibuppd,
--  weight_f,
--  height_f,
--  bmi_curr,
  age
FROM
  prostate_screening.prostate_overall
WHERE dx_psa > 0.0 AND
      (pros_gleason NOT IN (0.0, 99.0))
