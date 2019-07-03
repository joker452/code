(* Coursera Programming Languages, Homework 3, Provided Code *)

exception NoAnswer

datatype pattern = Wildcard
		 | Variable of string
		 | UnitP
		 | ConstP of int
		 | TupleP of pattern list
		 | ConstructorP of string * pattern

datatype valu = Const of int
	      | Unit
	      | Tuple of valu list
	      | Constructor of string * valu

fun g f1 f2 p =
    let 
	val r = g f1 f2 
    in
	case p of
	    Wildcard          => f1 ()
	  | Variable x        => f2 x
	  | TupleP ps         => List.foldl (fn (p,i) => (r p) + i) 0 ps
	  | ConstructorP(_,p) => r p
	  | _                 => 0
    end

val count_wildcards = g (fn () => 1) (fn s => 0)
val count_wild_and_variable_lengths = g (fn () => 1) (fn s => String.size s)
fun count_some_var(str_name, p) =
    g (fn () => 0) (fn s => if s = str_name then 1 else 0) p

fun check_pat(p) =
    let fun pat_vars(p) =
	    case p of
		Variable x => [x]
	      | TupleP ps => List.foldl (fn (p, vars) => pat_vars(p) @ vars) [] ps
	      | ConstructorP(_, p) => pat_vars(p)
	      | _ => []
	fun no_dup(str_list) =
	    case str_list of
		[] => true
	      | s::ss => if List.exists (fn s' => s' = s) ss then false else no_dup(ss)
    in (no_dup o pat_vars) p
    end

		 

(**** for the challenge problem only ****)

datatype typ = Anything
	     | UnitT
	     | IntT
	     | TupleT of typ list
	     | Datatype of string

(**** you can put all your code here ****)

fun only_capitals(str_list) =
    List.filter (fn str => Char.isUpper(String.sub (str, 0))) str_list

fun longest_string1(str_list) =
    foldl (fn (str, res_str) => if String.size str > String.size res_str then str else res_str) "" str_list

fun longest_string2(str_list) =
    foldl (fn (str, res_str) => if String.size str >= String.size res_str then str else res_str) "" str_list

fun longest_string_helper predicate str_list =
    foldl (fn (str, res_str) => if predicate (String.size str, String.size res_str) then str else res_str) "" str_list

val longest_string3 = longest_string_helper(fn (s1, s2) => s1 > s2)
val longest_string4 = longest_string_helper(fn (s1, s2) => s1 >= s2)

val longest_capitalized = longest_string1 o only_capitals 

val rev_string = String.implode o rev o String.explode

fun first_answer f l =
    case l of [] => raise NoAnswer
	    | x::xs => case f x of
			   SOME v => v
			 | NONE => first_answer f xs

fun all_answers f l =
    let fun answer_helper(f, l, res_list) =
	    case l of
		[] => SOME res_list
	      | x::xs => case f x of
			     SOME v => answer_helper(f, xs, res_list @ v)
			   | NONE => NONE
    in answer_helper(f, l, [])
    end

fun match(valu, pattern) =
    case (valu, pattern) of
	(_, Wildcard) => SOME []
      | (v, Variable s) => SOME [(s, v)]
      | (Unit, UnitP) => SOME []
      | (Const i, ConstP pi) => if i = pi then SOME [] else NONE
      | (Tuple vs, TupleP ps) => if List.length vs = List.length ps
				 then all_answers match (ListPair.zip (vs, ps))
				 else NONE
      | (Constructor(s2, v), ConstructorP(s1, p)) => if s1 = s2
						    then match(v, p)
						    else NONE
      | _ => NONE	
(* -> (string * valu) list option *)
fun first_match valu pattern_list =
    SOME (first_answer match (List.map (fn p => (valu, p)) pattern_list))
    handle NoAnswer => NONE
