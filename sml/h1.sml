(* (int * int * int) * (int * int * int) -> bool *)
fun is_older(date1: int * int * int, date2: int * int * int) =
    if (#1 date1) < (#1 date2)
    then true
    else if (#1 date1) = (#1 date2)
         then if (#2 date1) < (#2 date2)
	      then true
	      else if (#2 date1) = (#2 date2)
	           then if (#3 date1) < (#3 date2)
		        then true
			else false
	           else false
         else false

(* (int * int * int) list * int -> int *)
fun number_in_month(dates: (int * int * int) list, month: int) =
    if null dates
    then 0
    else let val tl_ans = number_in_month(tl dates, month)
	 in if #2 (hd dates) = month then tl_ans + 1 else tl_ans
	 end

(* (int * int * int) list * int list -> int *)
fun number_in_months(dates: (int * int * int) list, months: int list) =
    if null months
    then 0
    else number_in_month(dates, hd months) + number_in_months(dates, tl months)

(* (int * int * int) list * int * (int * int * int) list -> (int * int * int) list *)
fun dates_in_month(dates: (int * int * int) list, month: int) =
    if null dates
    then []
    else if #2 (hd dates) = month
         then (hd dates):: dates_in_month(tl dates, month)
         else dates_in_month(tl dates, month)

(* (int * int * int) list * int list -> (int * int * int) list *)
fun dates_in_months(dates: (int * int * int) list, months: int list) =
    if null months
    then []
    else dates_in_month(dates, hd months) @ dates_in_months(dates, tl months)

(* string list * int -> string *)
fun get_nth(strs: string list, n: int) =
    if n <= 0
    then ""
    else if n = 1 then hd strs else get_nth(tl strs, n - 1)

(* int * int * int -> string *)
fun date_to_string(date: int * int * int) =
    let val months = ["January", "February", "March", "April", "May", "June", "July",
		      "August", "September", "October", "November", "December"]
    in get_nth(months, #2 date) ^ " " ^ Int.toString(#3 date) ^ ", " ^ Int.toString(#1 date)
    end

(* int * int list -> int *)
fun number_before_reaching_sum(sum: int, nums: int list) =
    if sum - hd nums <= 0 then 0 else number_before_reaching_sum(sum - hd nums, tl nums) + 1

(* int -> int *)
fun what_month(day: int)=
    let val days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    in number_before_reaching_sum(day, days_in_month) + 1
    end

(* int * int -> int list *)
fun month_range(day1: int, day2: int) =
    let val m1 = what_month(day1)
	val m2 = what_month(day2)
	fun get_list(from: int, to: int, len: int) =
	    if len <= 1
	    then [to]
	    else
		if from < to then from:: get_list(from + 1, to, len - 1) else to:: get_list(from, to, len - 1)
    in if m2 < m1 then [] else get_list(m1, m2, day2 - day1 + 1)
    end

(* (int * int * int) list -> (int * int * int) option *)
fun oldest(dates: (int * int * int) list) =
    if null dates
    then NONE
    else
	let fun oldest_nonempty(dates: (int * int * int) list) =
	    if null (tl dates)
	    then hd dates
	    else let val tl_ans = oldest_nonempty(tl dates)
		 in
		     if is_older(hd dates, tl_ans)
		     then hd dates
		     else tl_ans
		 end
	in SOME (oldest_nonempty dates)
	end

(* int list -> int list *)
fun remove_duplicate(months: int list) =
    let fun contains(month: int, months: int list) =
	    if null months
	    then false
	    else if month = hd months then true else contains(month, tl months)
    in if null months
       then []
       else
	   if contains(hd months, tl months) then remove_duplicate(tl months) else hd months:: remove_duplicate(tl months)
    end
	
(* (int * int * int) list * int list -> int *)
fun number_in_months_challenge(dates: (int * int * int) list, months: int list) =
    number_in_months(dates, remove_duplicate(months))

(* (int * int * int) list * int list -> (int * int * int) list *)
fun dates_in_months_challenge(dates: (int * int * int) list, months: int list) =
    dates_in_months(dates, remove_duplicate(months))

(* (int * int * int) -> bool *)
fun reasonable_date(date: (int * int * int)) =
    if #1 date > 0
    then
	let val days_in_month = ["31", "~1", "31", "30", "31", "30", "31", "31", "30", "31", "30", "31"]
	in if #2 date = 2
	   then if #1 date mod 4 = 0 andalso #1 date mod 100 <> 0
		then if #3 date > 0 andalso #3 date < 30 then true else false
		else if #3 date > 0 andalso #3 date < 29 then true else false
	   else
	       if #2 date > 0 andalso #2 date < 13 andalso #3 date > 0
	       then if get_nth(days_in_month, #2 date)>= Int.toString(#3 date) then true else false
	       else false
	end
    else false
