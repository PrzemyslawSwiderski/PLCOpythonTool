SELECT
  #   po.plco_id,
  # PSA Closest To Diagnosis
  po.dx_psa,
  # Size of Gland - Sagittal
  sc.sizesag,
  # Size of Gland - Transverse
  sc.sizetran,
  # current BMI
  po.bmi_curr,
  po.height_f,
  # Number of packs smoked per day * years smoked.
  po.pack_years,
  po.educat,
  po.numbiopp,
  po.pros_is_first_dx,
  po.curative_radp,
  po.curative_hormp,
  po.curative_othp,
  po.curative_prostp,
  po.asp,
  po.ibup,
  po.age,
  # This is the Gleason score from a prostatectomy if available; otherwise it is the (more commonly available) score from the biopsy.
  po.pros_gleason,
  po.dth_days # Days from randomization until date of death.
FROM
  prostate_screening.prostate_overall po INNER JOIN prostate_screening.screening sc ON po.plco_id = sc.plco_id
WHERE dx_psa > 0.0 AND dx_psa < 40.0 AND sc.sizesag != '' AND sc.sizetran != '' AND is_dead = 1.0 AND
      (pros_gleason NOT IN (0.0, 99.0))
GROUP BY po.plco_id
# select * from prostate_overall po, screening sc where po.plco_id =sc.plco_id and dx_psa > 0.0 AND dx_psa < 40.0