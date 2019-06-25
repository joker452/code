(* Dan Grossman, Coursera PL, HW2 Provided Code *)

(* if you use this function to compare two strings (returns true if the same
   string), then you avoid several of the functions in problem 1 having
   polymorphic types that may be confusing *)
fun same_string(s1 : string, s2 : string) =
    s1 = s2

(* put your solutions for problem 1 here *)

(* you may assume that Num is always used with values 2, 3, ..., 10
   though it will not really come up *)
datatype suit = Clubs | Diamonds | Hearts | Spades
datatype rank = Jack | Queen | King | Ace | Num of int
type card = suit * rank

datatype color = Red | Black
datatype move = Discard of card | Draw 

exception IllegalMove

fun all_except_option(str, str_list) =
    let fun my_filter(str, str_list, res_list, contains)=
	    case str_list of
		[] => if contains then SOME (rev res_list) else NONE
	      | s::ss => if same_string(str, s)
			 then my_filter(str, ss, res_list, true)
			 else my_filter(str, ss, s:: res_list, contains)
    in my_filter(str, str_list, [], false)
    end

fun get_substitutions1(ss, str) =
    case ss of
	[] => []
      | s::ss' => case all_except_option(str, s) of
		      NONE => get_substitutions1(ss', str)
		    | SOME lst => lst @ get_substitutions1(ss', str)

fun get_substitutions2(ss, str) =
    let fun helper(ss, str, res_list) =
	    case ss of
		[] => res_list
	      | s::ss' => case all_except_option(str, s) of
			      NONE => helper(ss', str, res_list)
			    | SOME lst => helper(ss', str, res_list @ lst)
			  
    in helper(ss, str, [])
    end

fun similar_names(ss, full_name) =
    let fun change_firstname(new_firsts, partial_name, res_list) =
	    case new_firsts of
		[] => rev res_list
	      | f::fs => case partial_name of
			     {middle=y, last=z} => change_firstname(fs, partial_name, {first=f, middle=y, last=z}:: res_list)
    in case full_name of
	   {first=x, middle=y, last=z} => change_firstname(get_substitutions1(ss, x), {middle=y, last=z}, [full_name])
    end

fun card_color(card) =
    case card of
	(Clubs, _) => Black
      | (Spades, _) => Black
      | _ => Red

fun card_value(card) =
    case card of
	(_, Num n) => n
      | (_, Ace) => 11
      | _ => 10

fun remove_card(cards, card, e) =
    let fun remove_helper(cards, card, e, contains, res_list) =
	    case cards of
		[] => if contains then rev res_list else raise e
	      | c::cs => if c = card andalso contains = false
			 then remove_helper(cs, card, e, true, res_list)
			 else remove_helper(cs, card, e, contains, c::res_list)
    in remove_helper(cards, card, e, false, [])
    end
	
fun all_same_color(cards) =
    case cards of
	[] => true
      | [c] => true
      | c1::c2::cs => if card_color(c1) = card_color(c2) then all_same_color(c2::cs) else false

fun sum_cards(cards) =
    let fun sum_helper(cards, current_sum) =
	    case cards of
		[] => current_sum
	      | c::cs => sum_helper(cs, current_sum + card_value(c))
    in sum_helper(cards, 0)
    end

fun score(held_cards, goal) =
    let val sum = sum_cards(held_cards)
	val base_score = if sum > goal then 3 * (sum - goal) else goal - sum
    in if all_same_color(held_cards) then base_score div 2 else base_score
    end

fun officiate(cards, moves, goal) =
    let fun runner(cards, moves, goal, held_cards) =
	    case moves of
		[] => score(held_cards, goal) 
	      | Discard d::ms => runner(cards, ms, goal, remove_card(held_cards, d, IllegalMove))
	      | Draw::ms => case cards of
				[] => score(held_cards, goal)
			      | c::cs => if sum_cards(c:: held_cards) > goal
					 then score(c:: held_cards, goal)
					 else runner(cs, ms, goal, c:: held_cards)
    in runner(cards, moves, goal, [])
    end
