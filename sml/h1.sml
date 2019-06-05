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
    in get_nth(months, #2 date) ^ " " ^ Int.toString(#1 date) ^ ", " ^ Int.toString(#3 date)
    end
