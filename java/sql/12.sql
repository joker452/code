with good_attorneys(name, total_cases) as (select name, count(case_id) as total from attorneys join charges using (case_id) where name <> '' group by name having total > 100),
successful_cases(name, num_cases) as (select name, count(case_id) from attorneys join charges using (case_id) where disposition = 'Not Guilty'and name <> '' group by name),
cte as (select name, total_cases, (num_cases * 100.0 / total_cases) as rate
from good_attorneys join successful_cases using (name)
order by rate desc, total_cases desc
limit 7)
select a.name, a.total_cases, a.rate
from cte a, cte b
where a.rate < b.rate
group by a.name
having count(a.rate) = 6
limit 1;