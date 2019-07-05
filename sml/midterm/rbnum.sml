signature RBNUM =
sig
    type t
    val max_value: int
    val red_num: int -> t
    val blue_num: int -> t
    val is_blue: t -> bool
    val is_red: t -> bool
    val is_max_blue: t -> bool
    val to_int: t -> int
    exception OutOfRange
end

structure RBNum2 :> RBNUM =
struct
type t = int
exception OutOfRange
val max_value = 999
fun red_num i = if i > max_value orelse i < 0 then raise OutOfRange else i
fun blue_num i = if i > max_value orelse i < 0 then raise OutOfRange else i + 1000
(* int -> bool *)
fun is_blue x = x >= 1000
(* int -> bool *)
fun is_red x = x < 1000
(* int -> bool *)
fun is_max_blue x = x = 1999
(* int -> int *)
fun to_int x = if x > 999 then x - 1000 else x
end
(* (d) none of the function can be implemented equivalently outside the module by the client
(* is_max_blue can be implemented as
   fun is_max_blue x = if is_blue x andalso (to_int x) = max_value *)
*)
