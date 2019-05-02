static functions are functions that are only visible to other functions
in the same file (more precisely the same translation unit
a translation unit is the ultimate input to a C or C++ compiler
from which an object file is generated, or a source file after
it has been preprocessed.
"##" in "##__VA_ARGS__" make compiler ignore the "," before "##__VA_ARGS__"
when "##__VA_ARGS__" is empty

socket option SO_REUSEADDR should be set before bind() is called.

if listen socket is made non-blocking, then must use select/poll or loop to
make sure it is readble so a new connection can be accepted