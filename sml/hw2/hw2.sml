
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

fun replace_one_ace(cs) =
  case cs of
      [] => []
    | (c, Ace)::xs' => (c,Num(1))::xs'
    | card::xs' => card::replace_one_ace(xs')

fun replace_one_ace_move(cs) =
  case cs of
      [] => []
    | Discard(c, Ace)::xs' => Discard(c,Num(1))::xs'
    | card::xs' => card::replace_one_ace_move(xs')					

fun least_of(ls) =
  case ls of
      [] => 0
    | x::[] => x
    | x::xs' => let val min = least_of(xs')
		in if min < x
		   then min
		   else x
		end;
				     
fun score_challenge(cs,g) =
  let fun container(cs) =
	let val replaced = replace_one_ace(cs)
	in if cs = replaced
	   then [score(cs,g)]
	   else score(cs,g)::container(replaced)
	end
  in least_of(container(cs))
  end;

fun officiate_challenge(cds,moves,i) =
  let fun container(cs, moves) =
	let val replaced = replace_one_ace(cs)
	    val repl_moves = replace_one_ace_move(moves)
	in if cs = replaced
	   then [officiate(cs,moves,i)]
	   else officiate(cs,moves,i)::container(replaced, repl_moves)
	end
   in least_of(container(cds, moves))
  end;

fun discard_and_draw(held_cards, goal, held_cards_head,  move) =
    case held_cards of
	[] => move
      | c::cs => if score(held_cards_head @ cs, goal) = 0
		 then Discard c
		 else discard_and_draw(cs, goal, c::held_cards_head, move)
				      
fun careful_player(cards, goal) =
    let fun play_helper(cards, goal, held_cards, moves) =
	    if goal - sum_cards(held_cards) > 10
	    then case cards of
		     [] => rev (Draw:: moves)
		   | c::cs => play_helper(cs, goal, c::held_cards, Draw::moves)
	    else if score(held_cards, goal) = 0
	         then moves
	         else case cards of
			  [] => rev moves
			| c::cs => case discard_and_draw(held_cards, goal, [c], Draw) of
				       Discard c' => rev (Draw::Discard c'::moves)
			             | Draw => rev moves
    in play_helper(cards, goal,  [], [])
    end
