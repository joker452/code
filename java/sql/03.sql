select violation_county, count(case_id) as num_cases
from cases join charges using (case_id)
where violation_county <> '' and description like '%RECKLESS%'
group by violation_county
order by num_cases desc, violation_county asc
limit 3;