with valid_party(case_id, dob) as (select case_id, strftime('%Y.%m%d', dob) from parties where type = 'Defendant' and name <> '' and dob <> ''),
 valid_cases(case_id, filing_date, filing_year) as (select case_id, strftime('%Y.%m%d', filing_date), strftime('%Y', filing_date) from cases join charges using (case_id) where charges.disposition = 'Guilty' and filing_date <> '')
select filing_year, avg(cast (filing_date as float) - cast(dob as float))
from valid_party join valid_cases using (case_id)
where (cast (filing_year as float) - cast(dob as float))  > 0  and (cast (filing_year as float) - cast(dob as float))< 100
group by filing_year
order by filing_year desc
limit 5;