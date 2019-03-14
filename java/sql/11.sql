 with good_attorneys(name, total_cases) as (select name, count(case_id) as total from attorneys join charges using (case_id) where name <> '' group by name having total > 100),
successful_cases(name, num_cases) as (select name, count(case_id) from attorneys join charges using (case_id) where disposition = 'Not Guilty'and name <> '' group by name)
select name, total_cases, (num_cases * 100.0 / total_cases) as rate
from good_attorneys join successful_cases using (name)
order by rate desc, total_cases desc
limit 5;