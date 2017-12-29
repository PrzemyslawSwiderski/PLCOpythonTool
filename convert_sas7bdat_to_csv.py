from sas7bdat import SAS7BDAT

from config import *

with SAS7BDAT(f'%sfreepsa_data_feb16_d080516.sas7bdat' % FREE_PSA_PATH) as free_psa, \
        SAS7BDAT(f'%sprocp_data_feb16_d080516.sas7bdat' % DIAGNOSTIC_PROCEDURES_PATH) as diagnostic_procedures, \
        SAS7BDAT(f'%smedp_data_feb16_d080516.sas7bdat' % MEDICAL_COMPLICATIONS_PATH) as medical_complications, \
        SAS7BDAT(f'%sscreenp_data_feb16_d080516.sas7bdat' % SCREENING_PATH) as screening, \
        SAS7BDAT(f'%sscrsubp_data_feb16_d080516.sas7bdat' % SCREENING_ABNORMALITIES_PATH) as screening_abnormalities, \
        SAS7BDAT(f'%strt_p_data_feb16_d080516.sas7bdat' % TREATMENTS_PATH) as treatments, \
        SAS7BDAT(f'%spros_data_feb16_d080516.sas7bdat' % PROSTATE_PATH) as prostate_data:
    free_psa.convert_file(f'%sfreepsa_data_feb16_d080516.csv' % FREE_PSA_PATH)
    diagnostic_procedures.convert_file(f'%sprocp_data_feb16_d080516.csv' % DIAGNOSTIC_PROCEDURES_PATH)
    medical_complications.convert_file(f'%smedp_data_feb16_d080516.csv' % MEDICAL_COMPLICATIONS_PATH)
    screening.convert_file(f'%sscreenp_data_feb16_d080516.csv' % SCREENING_PATH)
    screening_abnormalities.convert_file(f'%sscrsubp_data_feb16_d080516.csv' % SCREENING_ABNORMALITIES_PATH)
    treatments.convert_file(f'%strt_p_data_feb16_d080516.csv' % TREATMENTS_PATH)
    prostate_data.convert_file(f'%spros_data_feb16_d080516.csv' % PROSTATE_PATH)
