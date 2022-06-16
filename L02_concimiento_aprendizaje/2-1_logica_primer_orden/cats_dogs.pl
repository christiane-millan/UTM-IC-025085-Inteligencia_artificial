
/* comentario
dos líneas */

dog(fido).
dog(rover). 
dog(henry). dog(tom).
cat(felix). cat(bill). cat(steve).
cat(michael).
cat(jane).
cat(mary).
cat(fido).
animal(X):-dog(X).
animal(Y):-cat(Y).
small(henry).
large(rover).
large(steve).
large(jane).
large(mike).
large(jim).
large_animal(X):-dog(X), large(X).
large_animal(Z):-cat(Z), large(Z).