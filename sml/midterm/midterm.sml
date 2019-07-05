datatype exp = Constant of int
	     | Double of exp
	     | Add of exp * exp
	     | Divide of exp * exp

(* exp -> int *)
fun eval_exp(e) =
    case e of
	Constant i => i
      | Double d => 2 * eval_exp(d)
      | Add (e1, e2) => eval_exp(e1) + eval_exp(e2)
      | Divide (e1, e2) => eval_exp(e1) div eval_exp(e2)

val div_by_zero = Divide (Constant 1, Add (Constant 1, Constant ~1))

(* exp -> bool *)
fun no_literal_zero_divide(e) =
    case e of
        Double e' => no_literal_zero_divide(e')
      | Add (e1, e2) => no_literal_zero_divide(e1) andalso no_literal_zero_divide(e2)
      | Divide (_, Constant 0) => false
      | Divide (e1, e2) => no_literal_zero_divide(e1) andalso no_literal_zero_divide(e2)
      | _ => true
		 
fun f (xs, ys) =
    case (xs, ys) of
	([], []) => SOME 0
      | (x::[], y::[]) => SOME (x + y)
      | (x1::x2::[], y1::y2::[]) => SOME (x1 + x2 + y1 + y2)
      | (x1::x2::xs', y1::y2::ys') => f (xs', ys')
      | _ => NONE

(*
(a) What is the type of f?
    int list * int list -> int option
(b) What does f([3], [10]) evaluate to?
    SOME 13
(c) What does f([3, 4], [10, 11]) evaluate to?
    SOME 28
(d) What does f([3, 4, 5], [10, 11, 12]) evaluate to?
    SOME 17
(e) What does f([3, 4, 5, 6], [10, 11, 12, 13]) evaluate to?
    SOME 35  (* SOME 36 *)
(f) Describe in at most 1 English sentence all the inputs to f such that the result of f is NONE.
    an tuple of int lists that have different length
(g) Yes or no: Is f tail-recursive?
    Yes
(h) What happens if we move branch 2 of f to be the first pattern in the case experssion?
    C
(i) What happens if we move branch 3 of f to be the first pattern in the case expression?
    C
(j) What happens if we move branch 4 of f to be the first pattern in the case expression?
    B  (* A, redundant branch is not allowed *)
(k) What happens if we move branch 5 of f to be the first pattern in the case expression?
    B  (* A, redundant branch is not allowed *)
*)

(*
fun f2 g xs =
    case xs of
	[] => []
      | x::xs' => (g x) :: f2 xs'  -> missing arguments

fun f3 xs =
    case xs of
	[] => NONE
      | x::[] => SOME 1
      | x::xs' => SOME (1 + (f3 xs')) -> this case branch doesn't have the same type as the first two branch

datatype t = A of int | B of (int * t) list
fun f4 x =
    let
	fun aux ys =
	    case ys of
		[] => []
	      | (i, j)::ys => (i + 1, j):: (aux ys)
    in
	case x of
	    A i => x
	  | B ys => B (aux x) -> branches in this case expr don't have the same type
end

exception Foo
fun f5 x = if x > 3 then x else raise Foo
fun f6 y = (f5 (y + 1)) handle _ => false -> this handle expr should have type int
*)

(* 'a * 'b -> 'c) -> 'a list -> 'b list -> 'c list *)
fun zipWith f xs ys =
    case (xs, ys) of
			([], _) => []
		  | (_, []) => []
	      | (x::xs', y::ys') => f((x, y)):: zipWith f xs' ys'

val first_bigger = zipWith (fn (x, y) => x > y)

fun zipWith' f xs ys =
    List.map f (ListPair.zip (xs, ys))

(*
(d) 3
(e) 6 											   
*)
			      
fun flat_map f xs =
    case xs of
	    [] => []
	  | x::xs' => (f x) @ flat_map f xs'

fun map f xs =
    flat_map (fn x => [f x]) xs
													 
fun filter f xs =
    flat_map (fn x => if f x then [x] else []) xs

													 
										  
    
