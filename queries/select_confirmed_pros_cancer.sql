select count(*) as 'not confirmed ill' from prostate_overall where confirmed_pros = '0';

select count(*) as 'confirmed ill' from prostate_overall where confirmed_pros = '1';

select count(*) as 'all' from prostate_overall;

select count(*) as 'confirmed ill with valid train_set' from prostate_overall where confirmed_pros = '1' AND (psa_level0!=0.0) AND (psa_level0<=35.0) AND (pros_gleason NOT IN (0.0, 99.0));

select * from prostate_overall where confirmed_pros = '1' AND f_cancersite = 1.0  AND (dx_psa!=0.0) AND (pros_gleason NOT IN (0.0, 99.0))