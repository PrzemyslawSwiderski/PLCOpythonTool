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
  curative_hormp,
  curative_othp,
  curative_prostp,
  curative_radp,
  rectal_history,
  confirmed_pros,
  surg_age,
  cig_years,
  numbiopp,
  age,
  asppd,
  ibuppd
--  bmi_curr,
FROM
  prostate_screening.prostate_overall
WHERE dx_psa > 0.0 AND
      (pros_gleason NOT IN (0.0, 99.0))
