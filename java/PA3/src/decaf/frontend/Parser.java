//### This file created by BYACC 1.8(/Java extension  1.13)
//### Java capabilities added 7 Jan 97, Bob Jamison
//### Updated : 27 Nov 97  -- Bob Jamison, Joe Nieten
//###           01 Jan 98  -- Bob Jamison -- fixed generic semantic constructor
//###           01 Jun 99  -- Bob Jamison -- added Runnable support
//###           06 Aug 00  -- Bob Jamison -- made state variables class-global
//###           03 Jan 01  -- Bob Jamison -- improved flags, tracing
//###           16 May 01  -- Bob Jamison -- added custom stack sizing
//###           04 Mar 02  -- Yuval Oren  -- improved java performance, added options
//###           14 Mar 02  -- Tomas Hurka -- -d support, static initializer workaround
//###           14 Sep 06  -- Keltin Leung-- ReduceListener support, eliminate underflow report in error recovery
//### Please send bug reports to tom@hukatronic.cz
//### static char yysccsid[] = "@(#)yaccpar	1.8 (Berkeley) 01/20/90";






//#line 11 "Parser.y"
package decaf.frontend;

import decaf.tree.Tree;
import decaf.tree.Tree.*;
import decaf.error.*;
import java.util.*;
//#line 25 "Parser.java"
interface ReduceListener {
  public boolean onReduce(String rule);
}




public class Parser
             extends BaseParser
             implements ReduceListener
{

boolean yydebug;        //do I want debug output?
int yynerrs;            //number of errors so far
int yyerrflag;          //was there an error?
int yychar;             //the current working character

ReduceListener reduceListener = null;
void yyclearin ()       {yychar = (-1);}
void yyerrok ()         {yyerrflag=0;}
void addReduceListener(ReduceListener l) {
  reduceListener = l;}


//########## MESSAGES ##########
//###############################################################
// method: debug
//###############################################################
void debug(String msg)
{
  if (yydebug)
    System.out.println(msg);
}

//########## STATE STACK ##########
final static int YYSTACKSIZE = 500;  //maximum stack size
int statestk[] = new int[YYSTACKSIZE]; //state stack
int stateptr;
int stateptrmax;                     //highest index of stackptr
int statemax;                        //state when highest index reached
//###############################################################
// methods: state stack push,pop,drop,peek
//###############################################################
final void state_push(int state)
{
  try {
		stateptr++;
		statestk[stateptr]=state;
	 }
	 catch (ArrayIndexOutOfBoundsException e) {
     int oldsize = statestk.length;
     int newsize = oldsize * 2;
     int[] newstack = new int[newsize];
     System.arraycopy(statestk,0,newstack,0,oldsize);
     statestk = newstack;
     statestk[stateptr]=state;
  }
}
final int state_pop()
{
  return statestk[stateptr--];
}
final void state_drop(int cnt)
{
  stateptr -= cnt; 
}
final int state_peek(int relative)
{
  return statestk[stateptr-relative];
}
//###############################################################
// method: init_stacks : allocate and prepare stacks
//###############################################################
final boolean init_stacks()
{
  stateptr = -1;
  val_init();
  return true;
}
//###############################################################
// method: dump_stacks : show n levels of the stacks
//###############################################################
void dump_stacks(int count)
{
int i;
  System.out.println("=index==state====value=     s:"+stateptr+"  v:"+valptr);
  for (i=0;i<count;i++)
    System.out.println(" "+i+"    "+statestk[i]+"      "+valstk[i]);
  System.out.println("======================");
}


//########## SEMANTIC VALUES ##########
//## **user defined:SemValue
String   yytext;//user variable to return contextual strings
SemValue yyval; //used to return semantic vals from action routines
SemValue yylval;//the 'lval' (result) I got from yylex()
SemValue valstk[] = new SemValue[YYSTACKSIZE];
int valptr;
//###############################################################
// methods: value stack push,pop,drop,peek.
//###############################################################
final void val_init()
{
  yyval=new SemValue();
  yylval=new SemValue();
  valptr=-1;
}
final void val_push(SemValue val)
{
  try {
    valptr++;
    valstk[valptr]=val;
  }
  catch (ArrayIndexOutOfBoundsException e) {
    int oldsize = valstk.length;
    int newsize = oldsize*2;
    SemValue[] newstack = new SemValue[newsize];
    System.arraycopy(valstk,0,newstack,0,oldsize);
    valstk = newstack;
    valstk[valptr]=val;
  }
}
final SemValue val_pop()
{
  return valstk[valptr--];
}
final void val_drop(int cnt)
{
  valptr -= cnt;
}
final SemValue val_peek(int relative)
{
  return valstk[valptr-relative];
}
//#### end semantic value section ####
public final static short VOID=257;
public final static short BOOL=258;
public final static short INT=259;
public final static short STRING=260;
public final static short CLASS=261;
public final static short NULL=262;
public final static short EXTENDS=263;
public final static short THIS=264;
public final static short WHILE=265;
public final static short FOR=266;
public final static short IF=267;
public final static short ELSE=268;
public final static short RETURN=269;
public final static short BREAK=270;
public final static short NEW=271;
public final static short PRINT=272;
public final static short READ_INTEGER=273;
public final static short READ_LINE=274;
public final static short LITERAL=275;
public final static short IDENTIFIER=276;
public final static short AND=277;
public final static short OR=278;
public final static short STATIC=279;
public final static short INSTANCEOF=280;
public final static short LESS_EQUAL=281;
public final static short GREATER_EQUAL=282;
public final static short EQUAL=283;
public final static short NOT_EQUAL=284;
public final static short SCOPY=285;
public final static short SEALED=286;
public final static short SEP=287;
public final static short VAR=288;
public final static short INIT=289;
public final static short DEFAULT=290;
public final static short UMINUS=291;
public final static short EMPTY=292;
public final static short YYERRCODE=256;
final static short yylhs[] = {                           -1,
    0,    1,    1,    3,    4,    5,    5,    5,    5,    5,
    5,    2,    2,    6,    6,    7,    7,    7,    9,    9,
   10,   10,    8,    8,   11,   12,   12,   13,   13,   13,
   13,   13,   13,   13,   13,   13,   13,   13,   21,   22,
   22,   24,   24,   25,   14,   14,   14,   28,   28,   26,
   26,   26,   29,   27,   23,   23,   23,   23,   23,   23,
   23,   23,   23,   23,   23,   23,   23,   23,   23,   23,
   23,   23,   23,   23,   23,   23,   23,   23,   23,   23,
   23,   23,   31,   31,   30,   30,   32,   32,   16,   17,
   20,   15,   33,   33,   18,   18,   19,
};
final static short yylen[] = {                            2,
    1,    2,    1,    2,    2,    1,    1,    1,    1,    2,
    3,    7,    6,    2,    0,    2,    2,    0,    1,    0,
    3,    1,    7,    6,    3,    2,    0,    1,    2,    1,
    1,    1,    2,    2,    2,    2,    1,    1,    6,    5,
    3,    3,    0,    3,    3,    1,    0,    2,    0,    2,
    4,    1,    2,    5,    1,    1,    1,    3,    3,    3,
    3,    3,    3,    3,    3,    3,    3,    6,    3,    3,
    3,    3,    3,    2,    2,    3,    3,    1,    4,    5,
    6,    5,    1,    1,    1,    0,    3,    1,    5,    9,
    1,    6,    2,    0,    2,    1,    4,
};
final static short yydefred[] = {                         0,
    0,    0,    0,    0,    3,    0,    0,    2,    0,    0,
    0,   14,   18,    0,    0,   18,    7,    8,    6,    9,
    0,    0,   13,   16,    0,    0,   17,    0,   10,    0,
    4,    0,    0,   12,    0,    0,   11,    0,   22,    0,
    0,    0,    0,    5,    0,    0,    0,   27,   24,   21,
   23,    0,   84,   78,    0,    0,    0,    0,   91,    0,
    0,    0,    0,   83,    0,    0,    0,    0,   25,    0,
    0,   28,   38,   26,    0,   30,   31,   32,    0,    0,
    0,    0,   37,    0,    0,    0,    0,   52,   57,    0,
    0,    0,    0,    0,   55,   56,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,   53,   29,   33,
   34,   35,   36,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,   48,    0,    0,    0,
    0,    0,    0,    0,   41,    0,    0,    0,    0,    0,
   76,   77,    0,    0,   73,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,   79,    0,
    0,   97,    0,    0,    0,    0,    0,    0,   89,    0,
    0,    0,   40,   42,   80,    0,    0,   82,    0,    0,
   54,    0,    0,   92,   44,   81,   39,   68,    0,   93,
    0,   90,
};
final static short yydgoto[] = {                          3,
    4,    5,   72,   25,   40,   10,   15,   27,   41,   42,
   73,   52,   74,   75,   76,   77,   78,   79,   80,   81,
   82,   83,   84,  136,  168,   95,   96,   87,   88,  177,
   89,  140,  194,
};
final static short yysindex[] = {                      -239,
 -252, -226,    0, -239,    0, -225, -207,    0, -206,  -86,
 -225,    0,    0,  -73,  272,    0,    0,    0,    0,    0,
 -202,  -85,    0,    0,   -8,  -90,    0,  300,    0,  -89,
    0,   39,  -18,    0,   47,  -85,    0,  -85,    0,  -87,
   48,   44,   52,    0,  -29,  -85,  -29,    0,    0,    0,
    0,   -1,    0,    0,   66,   72,  -23,  110,    0, -203,
   73,   78,   79,    0,   81,  110,  110,   51,    0,   85,
 -178,    0,    0,    0,   69,    0,    0,    0,   74,   75,
   77,   80,    0,  891,   68,    0, -145,    0,    0,  110,
  110,  110,    7,  891,    0,    0,   97,   49,  110,  100,
  101,  110,  -38,  -38, -132,  507, -131,    0,    0,    0,
    0,    0,    0,  110,  110,  110,  110,  110,  110,  110,
  110,  110,  110,  110,  110,  110,    0,  110,  110,  110,
  107,  531,   89,  558,    0,  110,  108,   70,  891,   24,
    0,    0,  582,  112,    0,  113,  934,  593,   35,   35,
  984,  984,   -6,   -6,  -38,  -38,  -38,   35,   35,  615,
  -32,  891,  110,   31,  110,   31,  708, -116,    0,  735,
  110,    0, -120,  110,  110, -129,  117,  115,    0,  841,
 -106,   31,    0,    0,    0,  891,  136,    0,  865,  110,
    0,  110,   31,    0,    0,    0,    0,    0,  137,    0,
   31,    0,
};
final static short yyrindex[] = {                         0,
    0,    0,    0,  179,    0,   57,    0,    0,    0,    0,
   57,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,  131,    0,    0,    0,  150,    0,  150,    0,    0,
    0,  158,    0,    0,    0,    0,    0,    0,    0,    0,
    0,  -53,    0,    0,    0,    0,    0,  -41,    0,    0,
    0,    0,    0,    0,    0,  -76,  -76,  -76,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,  912,  480,    0,    0,    0,  -76,
  -53,  -76,   90,  149,    0,    0,    0,    0,  -76,    0,
    0,  -76,  151,  409,    0,    0,    0,    0,    0,    0,
    0,    0,    0,  -76,  -76,  -76,  -76,  -76,  -76,  -76,
  -76,  -76,  -76,  -76,  -76,  -76,    0,  -76,  -76,  -76,
  123,    0,    0,    0,    0,  -76,    0,  -76,   42,    0,
    0,    0,    0,    0,    0,    0,    4,    2,  573,  882,
    8,   58,  759, 1006,  418,  445,  454,  964, 1014,    0,
 1033,  -25,  -14,  -53,  -76,  -53,    0,    0,    0,    0,
  -76,    0,    0,  -76,  -76,  160,    0,  171,    0,    0,
  -33,  -53,    0,    0,    0,   64,    0,    0,    0,  -76,
    0,    1,  -53,    0,    0,    0,    0,    0,    0,    0,
  -53,    0,
};
final static short yygindex[] = {                         0,
    0,  211,    5,  -17,   92,  206,  214,    0,  207,    0,
  -22,    0,  -55,  -88,    0,    0,    0,    0,    0,    0,
    0,    0, 1210,    0,    0,  896,  916,    0,    0,    0,
    0,   83,    0,
};
final static int YYTABLESIZE=1400;
static short yytable[];
static { yytable();}
static void yytable(){
yytable = new short[]{                         94,
   33,   33,  133,   33,  124,   47,   94,  127,  183,  122,
  120,   94,  121,  127,  123,   45,   92,   96,   39,   24,
   39,    1,   49,    6,   51,   94,   86,  126,   50,  125,
  124,   67,   24,   45,    7,  122,   13,    9,   68,  127,
  123,   47,   72,   66,   71,   72,    2,   71,   63,   16,
   31,   63,  128,   17,   18,   19,   20,   21,  128,   72,
   72,   71,   71,   67,  172,   63,   63,  171,   11,   12,
   68,  124,   97,   29,   37,   66,  122,  120,   36,  121,
  127,  123,   88,   67,  128,   88,   38,   46,   45,   94,
   68,   94,   47,   48,   72,   66,   71,  108,   64,   93,
   63,   64,   67,  199,   87,   90,   26,   87,  179,   68,
  181,   91,   99,   30,   66,   64,   64,  100,  101,   26,
  102,   48,   43,   69,  107,  128,  195,  109,  130,   43,
  131,  135,  110,  111,   43,  112,  137,  200,  113,  138,
  141,  142,   67,  144,  146,  202,  163,  165,  169,   68,
   64,   98,  174,   48,   66,  187,  175,  191,  171,   50,
  190,  193,   37,   50,   50,   50,   50,   50,   50,   50,
  184,   17,   18,   19,   20,   21,  196,  201,    1,   15,
   50,   50,   50,   50,   50,   32,   35,   74,   44,    5,
   20,   74,   74,   74,   74,   74,   51,   74,   19,   49,
   51,   51,   51,   51,   51,   51,   51,   95,   74,   74,
   74,   85,   74,   50,    8,   50,   14,   51,   51,   51,
   51,   51,   49,   94,   94,   94,   94,   94,   94,   28,
   94,   94,   94,   94,   49,   94,   94,   94,   94,   94,
   94,   94,   94,   74,   43,  178,   94,    0,  116,  117,
   51,   94,   51,   94,   94,   17,   18,   19,   20,   21,
   53,   49,   54,   55,   56,   57,    0,   58,   59,   60,
   61,   62,   63,   64,    0,    0,   49,    0,   65,   72,
   71,   71,    0,   70,   63,   63,   71,   17,   18,   19,
   20,   21,   53,    0,   54,   55,   56,   57,    0,   58,
   59,   60,   61,   62,   63,   64,    0,    0,    0,    0,
   65,  105,   53,    0,   54,   70,    0,    0,   71,    0,
    0,   60,    0,   62,   63,   64,    0,    0,    0,    0,
   65,   53,    0,   54,   64,   64,    0,    0,   71,    0,
   60,    0,   62,   63,   64,    0,    0,    0,    0,   65,
    0,   43,    0,   43,    0,    0,    0,   71,    0,    0,
   43,    0,   43,   43,   43,   43,    0,    0,    0,   43,
    0,   53,    0,   54,    0,    0,    0,   43,    0,    0,
   60,    0,   62,   63,   64,    0,    0,    0,    0,   65,
    0,    0,    0,    0,    0,    0,   23,   71,    0,   50,
   50,    0,    0,   50,   50,   50,   50,    0,    0,    0,
    0,   50,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,   34,    0,    0,   74,   74,    0,
    0,   74,   74,   74,   74,    0,   51,   51,    0,   74,
   51,   51,   51,   51,    0,   75,    0,    0,   51,   75,
   75,   75,   75,   75,   60,   75,    0,    0,   60,   60,
   60,   60,   60,    0,   60,    0,   75,   75,   75,    0,
   75,    0,    0,    0,    0,   60,   60,   60,    0,   60,
    0,   61,    0,    0,    0,   61,   61,   61,   61,   61,
   62,   61,    0,    0,   62,   62,   62,   62,   62,    0,
   62,   75,   61,   61,   61,    0,   61,    0,    0,    0,
   60,   62,   62,   62,    0,   62,   56,    0,    0,    0,
   46,   56,   56,    0,   56,   56,   56,    0,   17,   18,
   19,   20,   21,    0,    0,    0,    0,   61,   46,   56,
    0,   56,    0,  124,    0,    0,   62,  145,  122,  120,
   22,  121,  127,  123,    0,    0,   17,   18,   19,   20,
   21,    0,    0,    0,    0,    0,  126,  124,  125,    0,
   56,  164,  122,  120,    0,  121,  127,  123,   22,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
  126,    0,  125,    0,  124,    0,    0,  128,  166,  122,
  120,    0,  121,  127,  123,    0,    0,    0,    0,    0,
    0,    0,    0,   69,    0,    0,   69,  126,  124,  125,
    0,  128,    0,  122,  120,  173,  121,  127,  123,  124,
   69,   69,    0,    0,  122,  120,    0,  121,  127,  123,
    0,  126,    0,  125,    0,    0,    0,    0,  128,    0,
    0,  124,  126,    0,  125,    0,  122,  120,    0,  121,
  127,  123,    0,    0,    0,   69,    0,    0,    0,    0,
    0,    0,  128,    0,  126,    0,  125,    0,    0,    0,
    0,    0,    0,  128,    0,   75,   75,    0,    0,   75,
   75,   75,   75,    0,   60,   60,    0,   75,   60,   60,
   60,   60,    0,    0,    0,  128,   60,  176,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,   61,   61,    0,    0,   61,   61,   61,   61,    0,
   62,   62,    0,   61,   62,   62,   62,   62,    0,    0,
    0,    0,   62,    0,  124,    0,    0,    0,    0,  122,
  120,    0,  121,  127,  123,    0,   56,   56,    0,    0,
   56,   56,   56,   56,    0,  182,    0,  126,   56,  125,
    0,  124,    0,    0,    0,    0,  122,  120,    0,  121,
  127,  123,    0,  114,  115,    0,    0,  116,  117,  118,
  119,    0,    0,    0,  126,  129,  125,    0,  128,   58,
    0,   58,   58,   58,    0,    0,    0,  114,  115,    0,
    0,  116,  117,  118,  119,    0,   58,   58,   58,  129,
   58,    0,    0,    0,    0,  128,    0,  185,    0,    0,
    0,    0,    0,    0,  114,  115,    0,    0,  116,  117,
  118,  119,    0,    0,    0,    0,  129,    0,    0,   69,
   69,   58,    0,    0,    0,   69,   69,    0,  114,  115,
    0,   69,  116,  117,  118,  119,    0,    0,    0,  114,
  129,    0,    0,  116,  117,  118,  119,  124,    0,    0,
    0,  129,  122,  120,    0,  121,  127,  123,    0,    0,
    0,  114,  115,    0,    0,  116,  117,  118,  119,  192,
  126,  124,  125,  129,    0,  197,  122,  120,    0,  121,
  127,  123,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,   70,    0,  126,   70,  125,  124,    0,    0,
    0,  128,  122,  120,    0,  121,  127,  123,    0,   70,
   70,    0,    0,    0,    0,    0,    0,   85,   55,    0,
  126,    0,  125,   55,   55,  128,   55,   55,   55,    0,
    0,    0,    0,    0,    0,    0,    0,   86,    0,    0,
  124,   55,    0,   55,   70,  122,  120,    0,  121,  127,
  123,  128,    0,    0,  114,  115,   85,    0,  116,  117,
  118,  119,    0,  126,    0,  125,  129,    0,    0,    0,
    0,    0,   55,    0,   66,    0,   86,   66,    0,    0,
    0,  114,  115,    0,    0,  116,  117,  118,  119,    0,
  124,   66,   66,  129,  128,  122,  120,    0,  121,  127,
  123,    0,    0,    0,    0,   58,   58,    0,    0,   58,
   58,   58,   58,  126,    0,  125,   59,   58,   59,   59,
   59,    0,    0,    0,   65,    0,   66,   65,    0,   85,
    0,   85,    0,   59,   59,   59,    0,   59,    0,    0,
    0,   65,   65,   67,  128,    0,   67,   85,    0,   86,
    0,   86,    0,    0,    0,    0,    0,   85,   85,    0,
   67,   67,    0,    0,    0,    0,   85,   86,   59,    0,
    0,    0,    0,    0,    0,    0,   65,   86,   86,    0,
    0,    0,    0,    0,    0,    0,   86,  114,  115,    0,
    0,  116,  117,  118,  119,   67,    0,    0,    0,  129,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,  114,  115,    0,    0,  116,  117,  118,  119,    0,
    0,    0,    0,  129,    0,    0,    0,    0,   70,   70,
    0,    0,    0,    0,   70,   70,    0,  114,  115,    0,
   70,  116,  117,  118,  119,    0,    0,    0,    0,  129,
    0,    0,    0,    0,    0,    0,    0,    0,   55,   55,
    0,    0,   55,   55,   55,   55,    0,    0,    0,    0,
   55,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,  116,  117,  118,  119,    0,    0,
    0,    0,  129,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
   66,   66,    0,    0,    0,    0,   66,   66,    0,    0,
    0,    0,   66,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,  116,  117,    0,   94,    0,    0,
    0,    0,  129,    0,    0,  103,  104,  106,    0,    0,
    0,    0,   59,   59,    0,    0,   59,   59,   59,   59,
   65,   65,    0,    0,   59,    0,   65,   65,    0,  132,
    0,  134,   65,    0,    0,    0,    0,    0,  139,   67,
   67,  143,    0,    0,    0,   67,   67,    0,    0,    0,
    0,   67,    0,  147,  148,  149,  150,  151,  152,  153,
  154,  155,  156,  157,  158,  159,    0,  160,  161,  162,
    0,    0,    0,    0,    0,  167,    0,  170,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,  139,    0,  180,    0,    0,    0,    0,    0,
  186,    0,    0,  188,  189,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,  198,
};
}
static short yycheck[];
static { yycheck(); }
static void yycheck() {
yycheck = new short[] {                         33,
   91,   91,   91,   91,   37,   59,   40,   46,  125,   42,
   43,   45,   45,   46,   47,   41,   40,   59,   36,   15,
   38,  261,   45,  276,   47,   59,   41,   60,   46,   62,
   37,   33,   28,   59,  261,   42,  123,  263,   40,   46,
   47,   41,   41,   45,   41,   44,  286,   44,   41,  123,
   59,   44,   91,  257,  258,  259,  260,  261,   91,   58,
   59,   58,   59,   33,   41,   58,   59,   44,  276,  276,
   40,   37,  276,  276,   93,   45,   42,   43,   40,   45,
   46,   47,   41,   33,   91,   44,   40,   44,   41,  123,
   40,  125,   41,  123,   93,   45,   93,  276,   41,  123,
   93,   44,   33,  192,   41,   40,   15,   44,  164,   40,
  166,   40,   40,   22,   45,   58,   59,   40,   40,   28,
   40,  123,   33,  125,   40,   91,  182,   59,   61,   40,
  276,  125,   59,   59,   45,   59,   40,  193,   59,   91,
   41,   41,   33,  276,  276,  201,   40,   59,   41,   40,
   93,   60,   41,  123,   45,  276,   44,   41,   44,   37,
  290,  268,   93,   41,   42,   43,   44,   45,   46,   47,
  287,  257,  258,  259,  260,  261,   41,   41,    0,  123,
   58,   59,   60,   61,   62,  276,  276,   37,  276,   59,
   41,   41,   42,   43,   44,   45,   37,   47,   41,  276,
   41,   42,   43,   44,   45,   46,   47,   59,   58,   59,
   60,   41,   62,   91,    4,   93,   11,   58,   59,   60,
   61,   62,  276,  257,  258,  259,  260,  261,  262,   16,
  264,  265,  266,  267,  276,  269,  270,  271,  272,  273,
  274,  275,  276,   93,   38,  163,  280,   -1,  281,  282,
   91,  285,   93,  287,  288,  257,  258,  259,  260,  261,
  262,  276,  264,  265,  266,  267,   -1,  269,  270,  271,
  272,  273,  274,  275,   -1,   -1,  276,   -1,  280,  278,
  277,  278,   -1,  285,  277,  278,  288,  257,  258,  259,
  260,  261,  262,   -1,  264,  265,  266,  267,   -1,  269,
  270,  271,  272,  273,  274,  275,   -1,   -1,   -1,   -1,
  280,  261,  262,   -1,  264,  285,   -1,   -1,  288,   -1,
   -1,  271,   -1,  273,  274,  275,   -1,   -1,   -1,   -1,
  280,  262,   -1,  264,  277,  278,   -1,   -1,  288,   -1,
  271,   -1,  273,  274,  275,   -1,   -1,   -1,   -1,  280,
   -1,  262,   -1,  264,   -1,   -1,   -1,  288,   -1,   -1,
  271,   -1,  273,  274,  275,  276,   -1,   -1,   -1,  280,
   -1,  262,   -1,  264,   -1,   -1,   -1,  288,   -1,   -1,
  271,   -1,  273,  274,  275,   -1,   -1,   -1,   -1,  280,
   -1,   -1,   -1,   -1,   -1,   -1,  125,  288,   -1,  277,
  278,   -1,   -1,  281,  282,  283,  284,   -1,   -1,   -1,
   -1,  289,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,  125,   -1,   -1,  277,  278,   -1,
   -1,  281,  282,  283,  284,   -1,  277,  278,   -1,  289,
  281,  282,  283,  284,   -1,   37,   -1,   -1,  289,   41,
   42,   43,   44,   45,   37,   47,   -1,   -1,   41,   42,
   43,   44,   45,   -1,   47,   -1,   58,   59,   60,   -1,
   62,   -1,   -1,   -1,   -1,   58,   59,   60,   -1,   62,
   -1,   37,   -1,   -1,   -1,   41,   42,   43,   44,   45,
   37,   47,   -1,   -1,   41,   42,   43,   44,   45,   -1,
   47,   93,   58,   59,   60,   -1,   62,   -1,   -1,   -1,
   93,   58,   59,   60,   -1,   62,   37,   -1,   -1,   -1,
   41,   42,   43,   -1,   45,   46,   47,   -1,  257,  258,
  259,  260,  261,   -1,   -1,   -1,   -1,   93,   59,   60,
   -1,   62,   -1,   37,   -1,   -1,   93,   41,   42,   43,
  279,   45,   46,   47,   -1,   -1,  257,  258,  259,  260,
  261,   -1,   -1,   -1,   -1,   -1,   60,   37,   62,   -1,
   91,   41,   42,   43,   -1,   45,   46,   47,  279,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   60,   -1,   62,   -1,   37,   -1,   -1,   91,   41,   42,
   43,   -1,   45,   46,   47,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   41,   -1,   -1,   44,   60,   37,   62,
   -1,   91,   -1,   42,   43,   44,   45,   46,   47,   37,
   58,   59,   -1,   -1,   42,   43,   -1,   45,   46,   47,
   -1,   60,   -1,   62,   -1,   -1,   -1,   -1,   91,   -1,
   -1,   37,   60,   -1,   62,   -1,   42,   43,   -1,   45,
   46,   47,   -1,   -1,   -1,   93,   -1,   -1,   -1,   -1,
   -1,   -1,   91,   -1,   60,   -1,   62,   -1,   -1,   -1,
   -1,   -1,   -1,   91,   -1,  277,  278,   -1,   -1,  281,
  282,  283,  284,   -1,  277,  278,   -1,  289,  281,  282,
  283,  284,   -1,   -1,   -1,   91,  289,   93,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,  277,  278,   -1,   -1,  281,  282,  283,  284,   -1,
  277,  278,   -1,  289,  281,  282,  283,  284,   -1,   -1,
   -1,   -1,  289,   -1,   37,   -1,   -1,   -1,   -1,   42,
   43,   -1,   45,   46,   47,   -1,  277,  278,   -1,   -1,
  281,  282,  283,  284,   -1,   58,   -1,   60,  289,   62,
   -1,   37,   -1,   -1,   -1,   -1,   42,   43,   -1,   45,
   46,   47,   -1,  277,  278,   -1,   -1,  281,  282,  283,
  284,   -1,   -1,   -1,   60,  289,   62,   -1,   91,   41,
   -1,   43,   44,   45,   -1,   -1,   -1,  277,  278,   -1,
   -1,  281,  282,  283,  284,   -1,   58,   59,   60,  289,
   62,   -1,   -1,   -1,   -1,   91,   -1,   93,   -1,   -1,
   -1,   -1,   -1,   -1,  277,  278,   -1,   -1,  281,  282,
  283,  284,   -1,   -1,   -1,   -1,  289,   -1,   -1,  277,
  278,   93,   -1,   -1,   -1,  283,  284,   -1,  277,  278,
   -1,  289,  281,  282,  283,  284,   -1,   -1,   -1,  277,
  289,   -1,   -1,  281,  282,  283,  284,   37,   -1,   -1,
   -1,  289,   42,   43,   -1,   45,   46,   47,   -1,   -1,
   -1,  277,  278,   -1,   -1,  281,  282,  283,  284,   59,
   60,   37,   62,  289,   -1,   41,   42,   43,   -1,   45,
   46,   47,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   41,   -1,   60,   44,   62,   37,   -1,   -1,
   -1,   91,   42,   43,   -1,   45,   46,   47,   -1,   58,
   59,   -1,   -1,   -1,   -1,   -1,   -1,   52,   37,   -1,
   60,   -1,   62,   42,   43,   91,   45,   46,   47,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   52,   -1,   -1,
   37,   60,   -1,   62,   93,   42,   43,   -1,   45,   46,
   47,   91,   -1,   -1,  277,  278,   91,   -1,  281,  282,
  283,  284,   -1,   60,   -1,   62,  289,   -1,   -1,   -1,
   -1,   -1,   91,   -1,   41,   -1,   91,   44,   -1,   -1,
   -1,  277,  278,   -1,   -1,  281,  282,  283,  284,   -1,
   37,   58,   59,  289,   91,   42,   43,   -1,   45,   46,
   47,   -1,   -1,   -1,   -1,  277,  278,   -1,   -1,  281,
  282,  283,  284,   60,   -1,   62,   41,  289,   43,   44,
   45,   -1,   -1,   -1,   41,   -1,   93,   44,   -1,  164,
   -1,  166,   -1,   58,   59,   60,   -1,   62,   -1,   -1,
   -1,   58,   59,   41,   91,   -1,   44,  182,   -1,  164,
   -1,  166,   -1,   -1,   -1,   -1,   -1,  192,  193,   -1,
   58,   59,   -1,   -1,   -1,   -1,  201,  182,   93,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   93,  192,  193,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,  201,  277,  278,   -1,
   -1,  281,  282,  283,  284,   93,   -1,   -1,   -1,  289,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,  277,  278,   -1,   -1,  281,  282,  283,  284,   -1,
   -1,   -1,   -1,  289,   -1,   -1,   -1,   -1,  277,  278,
   -1,   -1,   -1,   -1,  283,  284,   -1,  277,  278,   -1,
  289,  281,  282,  283,  284,   -1,   -1,   -1,   -1,  289,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,  277,  278,
   -1,   -1,  281,  282,  283,  284,   -1,   -1,   -1,   -1,
  289,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,  281,  282,  283,  284,   -1,   -1,
   -1,   -1,  289,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
  277,  278,   -1,   -1,   -1,   -1,  283,  284,   -1,   -1,
   -1,   -1,  289,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,  281,  282,   -1,   58,   -1,   -1,
   -1,   -1,  289,   -1,   -1,   66,   67,   68,   -1,   -1,
   -1,   -1,  277,  278,   -1,   -1,  281,  282,  283,  284,
  277,  278,   -1,   -1,  289,   -1,  283,  284,   -1,   90,
   -1,   92,  289,   -1,   -1,   -1,   -1,   -1,   99,  277,
  278,  102,   -1,   -1,   -1,  283,  284,   -1,   -1,   -1,
   -1,  289,   -1,  114,  115,  116,  117,  118,  119,  120,
  121,  122,  123,  124,  125,  126,   -1,  128,  129,  130,
   -1,   -1,   -1,   -1,   -1,  136,   -1,  138,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,  163,   -1,  165,   -1,   -1,   -1,   -1,   -1,
  171,   -1,   -1,  174,  175,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,  190,
};
}
final static short YYFINAL=3;
final static short YYMAXTOKEN=292;
final static String yyname[] = {
"end-of-file",null,null,null,null,null,null,null,null,null,null,null,null,null,
null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,
null,null,null,"'!'",null,null,null,"'%'",null,null,"'('","')'","'*'","'+'",
"','","'-'","'.'","'/'",null,null,null,null,null,null,null,null,null,null,"':'",
"';'","'<'","'='","'>'",null,null,null,null,null,null,null,null,null,null,null,
null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,
null,"'['",null,"']'",null,null,null,null,null,null,null,null,null,null,null,
null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,
null,null,"'{'",null,"'}'",null,null,null,null,null,null,null,null,null,null,
null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,
null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,
null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,
null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,
null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,
null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,
null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,
null,null,null,null,null,null,null,null,null,"VOID","BOOL","INT","STRING",
"CLASS","NULL","EXTENDS","THIS","WHILE","FOR","IF","ELSE","RETURN","BREAK",
"NEW","PRINT","READ_INTEGER","READ_LINE","LITERAL","IDENTIFIER","AND","OR",
"STATIC","INSTANCEOF","LESS_EQUAL","GREATER_EQUAL","EQUAL","NOT_EQUAL","SCOPY",
"SEALED","SEP","VAR","INIT","DEFAULT","UMINUS","EMPTY",
};
final static String yyrule[] = {
"$accept : Program",
"Program : ClassList",
"ClassList : ClassList ClassDef",
"ClassList : ClassDef",
"VariableDef : Variable ';'",
"Variable : Type IDENTIFIER",
"Type : INT",
"Type : VOID",
"Type : BOOL",
"Type : STRING",
"Type : CLASS IDENTIFIER",
"Type : Type '[' ']'",
"ClassDef : SEALED CLASS IDENTIFIER ExtendsClause '{' FieldList '}'",
"ClassDef : CLASS IDENTIFIER ExtendsClause '{' FieldList '}'",
"ExtendsClause : EXTENDS IDENTIFIER",
"ExtendsClause :",
"FieldList : FieldList VariableDef",
"FieldList : FieldList FunctionDef",
"FieldList :",
"Formals : VariableList",
"Formals :",
"VariableList : VariableList ',' Variable",
"VariableList : Variable",
"FunctionDef : STATIC Type IDENTIFIER '(' Formals ')' StmtBlock",
"FunctionDef : Type IDENTIFIER '(' Formals ')' StmtBlock",
"StmtBlock : '{' StmtList '}'",
"StmtList : StmtList Stmt",
"StmtList :",
"Stmt : VariableDef",
"Stmt : SimpleStmt ';'",
"Stmt : IfStmt",
"Stmt : WhileStmt",
"Stmt : ForStmt",
"Stmt : ReturnStmt ';'",
"Stmt : PrintStmt ';'",
"Stmt : BreakStmt ';'",
"Stmt : OCStmt ';'",
"Stmt : GuardedStmt",
"Stmt : StmtBlock",
"OCStmt : SCOPY '(' IDENTIFIER ',' Expr ')'",
"GuardedStmt : IF '{' GuardList Guard '}'",
"GuardedStmt : IF '{' '}'",
"GuardList : GuardList Guard SEP",
"GuardList :",
"Guard : Expr ':' Stmt",
"SimpleStmt : LValue '=' Expr",
"SimpleStmt : Call",
"SimpleStmt :",
"Receiver : Expr '.'",
"Receiver :",
"LValue : Receiver IDENTIFIER",
"LValue : Expr '[' Expr ']'",
"LValue : AutoVariable",
"AutoVariable : VAR IDENTIFIER",
"Call : Receiver IDENTIFIER '(' Actuals ')'",
"Expr : LValue",
"Expr : Call",
"Expr : Constant",
"Expr : Expr '+' Expr",
"Expr : Expr '-' Expr",
"Expr : Expr '*' Expr",
"Expr : Expr '/' Expr",
"Expr : Expr '%' Expr",
"Expr : Expr EQUAL Expr",
"Expr : Expr NOT_EQUAL Expr",
"Expr : Expr '<' Expr",
"Expr : Expr '>' Expr",
"Expr : Expr INIT Expr",
"Expr : Expr '[' Expr ']' DEFAULT Expr",
"Expr : Expr LESS_EQUAL Expr",
"Expr : Expr GREATER_EQUAL Expr",
"Expr : Expr AND Expr",
"Expr : Expr OR Expr",
"Expr : '(' Expr ')'",
"Expr : '-' Expr",
"Expr : '!' Expr",
"Expr : READ_INTEGER '(' ')'",
"Expr : READ_LINE '(' ')'",
"Expr : THIS",
"Expr : NEW IDENTIFIER '(' ')'",
"Expr : NEW Type '[' Expr ']'",
"Expr : INSTANCEOF '(' Expr ',' IDENTIFIER ')'",
"Expr : '(' CLASS IDENTIFIER ')' Expr",
"Constant : LITERAL",
"Constant : NULL",
"Actuals : ExprList",
"Actuals :",
"ExprList : ExprList ',' Expr",
"ExprList : Expr",
"WhileStmt : WHILE '(' Expr ')' Stmt",
"ForStmt : FOR '(' SimpleStmt ';' Expr ';' SimpleStmt ')' Stmt",
"BreakStmt : BREAK",
"IfStmt : IF '(' Expr ')' Stmt ElseClause",
"ElseClause : ELSE Stmt",
"ElseClause :",
"ReturnStmt : RETURN Expr",
"ReturnStmt : RETURN",
"PrintStmt : PRINT '(' ExprList ')'",
};

//#line 480 "Parser.y"
    
	/**
	 * 打印当前归约所用的语法规则<br>
	 * 请勿修改。
	 */
    public boolean onReduce(String rule) {
		if (rule.startsWith("$$"))
			return false;
		else
			rule = rule.replaceAll(" \\$\\$\\d+", "");

   	    if (rule.endsWith(":"))
    	    System.out.println(rule + " <empty>");
   	    else
			System.out.println(rule);
		return false;
    }
    
    public void diagnose() {
		addReduceListener(this);
		yyparse();
	}
//#line 688 "Parser.java"
//###############################################################
// method: yylexdebug : check lexer state
//###############################################################
void yylexdebug(int state,int ch)
{
String s=null;
  if (ch < 0) ch=0;
  if (ch <= YYMAXTOKEN) //check index bounds
     s = yyname[ch];    //now get it
  if (s==null)
    s = "illegal-symbol";
  debug("state "+state+", reading "+ch+" ("+s+")");
}





//The following are now global, to aid in error reporting
int yyn;       //next next thing to do
int yym;       //
int yystate;   //current parsing state from state table
String yys;    //current token string


//###############################################################
// method: yyparse : parse input and execute indicated items
//###############################################################
int yyparse()
{
boolean doaction;
  init_stacks();
  yynerrs = 0;
  yyerrflag = 0;
  yychar = -1;          //impossible char forces a read
  yystate=0;            //initial state
  state_push(yystate);  //save it
  while (true) //until parsing is done, either correctly, or w/error
    {
    doaction=true;
    //if (yydebug) debug("loop"); 
    //#### NEXT ACTION (from reduction table)
    for (yyn=yydefred[yystate];yyn==0;yyn=yydefred[yystate])
      {
      //if (yydebug) debug("yyn:"+yyn+"  state:"+yystate+"  yychar:"+yychar);
      if (yychar < 0)      //we want a char?
        {
        yychar = yylex();  //get next token
        //if (yydebug) debug(" next yychar:"+yychar);
        //#### ERROR CHECK ####
        //if (yychar < 0)    //it it didn't work/error
        //  {
        //  yychar = 0;      //change it to default string (no -1!)
          //if (yydebug)
          //  yylexdebug(yystate,yychar);
        //  }
        }//yychar<0
      yyn = yysindex[yystate];  //get amount to shift by (shift index)
      if ((yyn != 0) && (yyn += yychar) >= 0 &&
          yyn <= YYTABLESIZE && yycheck[yyn] == yychar)
        {
        //if (yydebug)
          //debug("state "+yystate+", shifting to state "+yytable[yyn]);
        //#### NEXT STATE ####
        yystate = yytable[yyn];//we are in a new state
        state_push(yystate);   //save it
        val_push(yylval);      //push our lval as the input for next rule
        yychar = -1;           //since we have 'eaten' a token, say we need another
        if (yyerrflag > 0)     //have we recovered an error?
           --yyerrflag;        //give ourselves credit
        doaction=false;        //but don't process yet
        break;   //quit the yyn=0 loop
        }

    yyn = yyrindex[yystate];  //reduce
    if ((yyn !=0 ) && (yyn += yychar) >= 0 &&
            yyn <= YYTABLESIZE && yycheck[yyn] == yychar)
      {   //we reduced!
      //if (yydebug) debug("reduce");
      yyn = yytable[yyn];
      doaction=true; //get ready to execute
      break;         //drop down to actions
      }
    else //ERROR RECOVERY
      {
      if (yyerrflag==0)
        {
        yyerror("syntax error");
        yynerrs++;
        }
      if (yyerrflag < 3) //low error count?
        {
        yyerrflag = 3;
        while (true)   //do until break
          {
          if (stateptr<0 || valptr<0)   //check for under & overflow here
            {
            return 1;
            }
          yyn = yysindex[state_peek(0)];
          if ((yyn != 0) && (yyn += YYERRCODE) >= 0 &&
                    yyn <= YYTABLESIZE && yycheck[yyn] == YYERRCODE)
            {
            //if (yydebug)
              //debug("state "+state_peek(0)+", error recovery shifting to state "+yytable[yyn]+" ");
            yystate = yytable[yyn];
            state_push(yystate);
            val_push(yylval);
            doaction=false;
            break;
            }
          else
            {
            //if (yydebug)
              //debug("error recovery discarding state "+state_peek(0)+" ");
            if (stateptr<0 || valptr<0)   //check for under & overflow here
              {
              return 1;
              }
            state_pop();
            val_pop();
            }
          }
        }
      else            //discard this token
        {
        if (yychar == 0)
          return 1; //yyabort
        //if (yydebug)
          //{
          //yys = null;
          //if (yychar <= YYMAXTOKEN) yys = yyname[yychar];
          //if (yys == null) yys = "illegal-symbol";
          //debug("state "+yystate+", error recovery discards token "+yychar+" ("+yys+")");
          //}
        yychar = -1;  //read another
        }
      }//end error recovery
    }//yyn=0 loop
    if (!doaction)   //any reason not to proceed?
      continue;      //skip action
    yym = yylen[yyn];          //get count of terminals on rhs
    //if (yydebug)
      //debug("state "+yystate+", reducing "+yym+" by rule "+yyn+" ("+yyrule[yyn]+")");
    if (yym>0)                 //if count of rhs not 'nil'
      yyval = val_peek(yym-1); //get current semantic value
    if (reduceListener == null || reduceListener.onReduce(yyrule[yyn])) // if intercepted!
      switch(yyn)
      {
//########## USER-SUPPLIED ACTIONS ##########
case 1:
//#line 54 "Parser.y"
{
						tree = new Tree.TopLevel(val_peek(0).clist, val_peek(0).loc);
					}
break;
case 2:
//#line 60 "Parser.y"
{
						yyval.clist.add(val_peek(0).cdef);
					}
break;
case 3:
//#line 64 "Parser.y"
{
                		yyval.clist = new ArrayList<Tree.ClassDef>();
                		yyval.clist.add(val_peek(0).cdef);
                	}
break;
case 5:
//#line 74 "Parser.y"
{
						yyval.vdef = new Tree.VarDef(val_peek(0).ident, val_peek(1).type, val_peek(0).loc);
					}
break;
case 6:
//#line 80 "Parser.y"
{
						yyval.type = new Tree.TypeIdent(Tree.INT, val_peek(0).loc);
					}
break;
case 7:
//#line 84 "Parser.y"
{
                		yyval.type = new Tree.TypeIdent(Tree.VOID, val_peek(0).loc);
                	}
break;
case 8:
//#line 88 "Parser.y"
{
                		yyval.type = new Tree.TypeIdent(Tree.BOOL, val_peek(0).loc);
                	}
break;
case 9:
//#line 92 "Parser.y"
{
                		yyval.type = new Tree.TypeIdent(Tree.STRING, val_peek(0).loc);
                	}
break;
case 10:
//#line 96 "Parser.y"
{
                		yyval.type = new Tree.TypeClass(val_peek(0).ident, val_peek(1).loc);
                	}
break;
case 11:
//#line 100 "Parser.y"
{
                		yyval.type = new Tree.TypeArray(val_peek(2).type, val_peek(2).loc);
                	}
break;
case 12:
//#line 106 "Parser.y"
{
						yyval.cdef = new Tree.ClassDef(true, val_peek(4).ident, val_peek(3).ident, val_peek(1).flist, val_peek(5).loc);
					}
break;
case 13:
//#line 110 "Parser.y"
{
						yyval.cdef = new Tree.ClassDef(false, val_peek(4).ident, val_peek(3).ident, val_peek(1).flist, val_peek(5).loc);
					}
break;
case 14:
//#line 116 "Parser.y"
{
						yyval.ident = val_peek(0).ident;
					}
break;
case 15:
//#line 120 "Parser.y"
{
                		yyval = new SemValue();
                	}
break;
case 16:
//#line 126 "Parser.y"
{
						yyval.flist.add(val_peek(0).vdef);
					}
break;
case 17:
//#line 130 "Parser.y"
{
						yyval.flist.add(val_peek(0).fdef);
					}
break;
case 18:
//#line 134 "Parser.y"
{
                		yyval = new SemValue();
                		yyval.flist = new ArrayList<Tree>();
                	}
break;
case 20:
//#line 142 "Parser.y"
{
                		yyval = new SemValue();
                		yyval.vlist = new ArrayList<Tree.VarDef>(); 
                	}
break;
case 21:
//#line 149 "Parser.y"
{
						yyval.vlist.add(val_peek(0).vdef);
					}
break;
case 22:
//#line 153 "Parser.y"
{
                		yyval.vlist = new ArrayList<Tree.VarDef>();
						yyval.vlist.add(val_peek(0).vdef);
                	}
break;
case 23:
//#line 160 "Parser.y"
{
						yyval.fdef = new MethodDef(true, val_peek(4).ident, val_peek(5).type, val_peek(2).vlist, (Block) val_peek(0).stmt, val_peek(4).loc);
					}
break;
case 24:
//#line 164 "Parser.y"
{
						yyval.fdef = new MethodDef(false, val_peek(4).ident, val_peek(5).type, val_peek(2).vlist, (Block) val_peek(0).stmt, val_peek(4).loc);
					}
break;
case 25:
//#line 170 "Parser.y"
{
						yyval.stmt = new Block(val_peek(1).slist, val_peek(2).loc);
					}
break;
case 26:
//#line 176 "Parser.y"
{
						yyval.slist.add(val_peek(0).stmt);
					}
break;
case 27:
//#line 180 "Parser.y"
{
                		yyval = new SemValue();
                		yyval.slist = new ArrayList<Tree>();
                	}
break;
case 28:
//#line 187 "Parser.y"
{
						yyval.stmt = val_peek(0).vdef;
					}
break;
case 29:
//#line 192 "Parser.y"
{
                		if (yyval.stmt == null) {
                			yyval.stmt = new Tree.Skip(val_peek(0).loc);
                		}
                	}
break;
case 39:
//#line 209 "Parser.y"
{
						yyval.stmt = new Tree.Scopy(val_peek(1).expr, val_peek(3).ident, val_peek(5).loc);
					}
break;
case 40:
//#line 215 "Parser.y"
{
                        val_peek(2).slist.add(val_peek(1).stmt);
                        yyval.stmt = new Tree.GuardStmt(val_peek(2).slist, val_peek(4).loc);
                    }
break;
case 41:
//#line 220 "Parser.y"
{
				        yyval.stmt = new Tree.GuardStmt(null, val_peek(2).loc);
				    }
break;
case 42:
//#line 226 "Parser.y"
{
                       yyval.slist.add(val_peek(1).stmt);
                    }
break;
case 43:
//#line 230 "Parser.y"
{
				    	yyval = new SemValue();
				    	yyval.slist = new ArrayList<Tree>();
				    }
break;
case 44:
//#line 237 "Parser.y"
{
						yyval.stmt = new Tree.Guard(val_peek(2).expr, val_peek(0).stmt, val_peek(2).loc);
					}
break;
case 45:
//#line 243 "Parser.y"
{
						yyval.stmt = new Tree.Assign(val_peek(2).lvalue, val_peek(0).expr, val_peek(1).loc);
					}
break;
case 46:
//#line 247 "Parser.y"
{
                		yyval.stmt = new Tree.Exec(val_peek(0).expr, val_peek(0).loc);
                	}
break;
case 47:
//#line 251 "Parser.y"
{
                		yyval = new SemValue();
                	}
break;
case 49:
//#line 258 "Parser.y"
{
                		yyval = new SemValue();
                	}
break;
case 50:
//#line 264 "Parser.y"
{
						yyval.lvalue = new Tree.Ident(false, val_peek(1).expr, val_peek(0).ident, val_peek(0).loc);
						if (val_peek(1).loc == null) {
							yyval.loc = val_peek(0).loc;
						}
					}
break;
case 51:
//#line 271 "Parser.y"
{
                		yyval.lvalue = new Tree.Indexed(val_peek(3).expr, val_peek(1).expr, val_peek(3).loc);
                	}
break;
case 53:
//#line 278 "Parser.y"
{
                    	yyval.lvalue = new Tree.Ident(true, null, val_peek(0).ident, val_peek(0).loc);
                    }
break;
case 54:
//#line 284 "Parser.y"
{
						yyval.expr = new Tree.CallExpr(val_peek(4).expr, val_peek(3).ident, val_peek(1).elist, val_peek(3).loc);
						if (val_peek(4).loc == null) {
							yyval.loc = val_peek(3).loc;
						}
					}
break;
case 55:
//#line 293 "Parser.y"
{
						yyval.expr = val_peek(0).lvalue;
					}
break;
case 58:
//#line 299 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.PLUS, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 59:
//#line 303 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.MINUS, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 60:
//#line 307 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.MUL, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 61:
//#line 311 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.DIV, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 62:
//#line 315 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.MOD, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 63:
//#line 319 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.EQ, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 64:
//#line 323 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.NE, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 65:
//#line 327 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.LT, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 66:
//#line 331 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.GT, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 67:
//#line 335 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.INIT, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 68:
//#line 339 "Parser.y"
{
                		yyval.expr = new Tree.Default(val_peek(5).expr, val_peek(3).expr, val_peek(0).expr, val_peek(5).loc);
                	}
break;
case 69:
//#line 343 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.LE, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 70:
//#line 347 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.GE, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 71:
//#line 351 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.AND, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 72:
//#line 355 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.OR, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 73:
//#line 359 "Parser.y"
{
                		yyval = val_peek(1);
                	}
break;
case 74:
//#line 363 "Parser.y"
{
                		yyval.expr = new Tree.Unary(Tree.NEG, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 75:
//#line 367 "Parser.y"
{
                		yyval.expr = new Tree.Unary(Tree.NOT, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 76:
//#line 371 "Parser.y"
{
                		yyval.expr = new Tree.ReadIntExpr(val_peek(2).loc);
                	}
break;
case 77:
//#line 375 "Parser.y"
{
                		yyval.expr = new Tree.ReadLineExpr(val_peek(2).loc);
                	}
break;
case 78:
//#line 379 "Parser.y"
{
                		yyval.expr = new Tree.ThisExpr(val_peek(0).loc);
                	}
break;
case 79:
//#line 383 "Parser.y"
{
                		yyval.expr = new Tree.NewClass(val_peek(2).ident, val_peek(3).loc);
                	}
break;
case 80:
//#line 387 "Parser.y"
{
                		yyval.expr = new Tree.NewArray(val_peek(3).type, val_peek(1).expr, val_peek(4).loc);
                	}
break;
case 81:
//#line 391 "Parser.y"
{
                		yyval.expr = new Tree.TypeTest(val_peek(3).expr, val_peek(1).ident, val_peek(5).loc);
                	}
break;
case 82:
//#line 395 "Parser.y"
{
                		yyval.expr = new Tree.TypeCast(val_peek(2).ident, val_peek(0).expr, val_peek(0).loc);
                	}
break;
case 83:
//#line 401 "Parser.y"
{
						yyval.expr = new Tree.Literal(val_peek(0).typeTag, val_peek(0).literal, val_peek(0).loc);
					}
break;
case 84:
//#line 405 "Parser.y"
{
						yyval.expr = new Null(val_peek(0).loc);
					}
break;
case 86:
//#line 412 "Parser.y"
{
                		yyval = new SemValue();
                		yyval.elist = new ArrayList<Tree.Expr>();
                	}
break;
case 87:
//#line 419 "Parser.y"
{
						yyval.elist.add(val_peek(0).expr);
					}
break;
case 88:
//#line 423 "Parser.y"
{
                		yyval.elist = new ArrayList<Tree.Expr>();
						yyval.elist.add(val_peek(0).expr);
                	}
break;
case 89:
//#line 430 "Parser.y"
{
						yyval.stmt = new Tree.WhileLoop(val_peek(2).expr, val_peek(0).stmt, val_peek(4).loc);
					}
break;
case 90:
//#line 436 "Parser.y"
{
						yyval.stmt = new Tree.ForLoop(val_peek(6).stmt, val_peek(4).expr, val_peek(2).stmt, val_peek(0).stmt, val_peek(8).loc);
					}
break;
case 91:
//#line 442 "Parser.y"
{
						yyval.stmt = new Tree.Break(val_peek(0).loc);
					}
break;
case 92:
//#line 448 "Parser.y"
{
						yyval.stmt = new Tree.If(val_peek(3).expr, val_peek(1).stmt, val_peek(0).stmt, val_peek(5).loc);
					}
break;
case 93:
//#line 454 "Parser.y"
{
						yyval.stmt = val_peek(0).stmt;
					}
break;
case 94:
//#line 458 "Parser.y"
{
						yyval = new SemValue();
					}
break;
case 95:
//#line 464 "Parser.y"
{
						yyval.stmt = new Tree.Return(val_peek(0).expr, val_peek(1).loc);
					}
break;
case 96:
//#line 468 "Parser.y"
{
                		yyval.stmt = new Tree.Return(null, val_peek(0).loc);
                	}
break;
case 97:
//#line 474 "Parser.y"
{
						yyval.stmt = new Print(val_peek(1).elist, val_peek(3).loc);
					}
break;
//#line 1337 "Parser.java"
//########## END OF USER-SUPPLIED ACTIONS ##########
    }//switch
    //#### Now let's reduce... ####
    //if (yydebug) debug("reduce");
    state_drop(yym);             //we just reduced yylen states
    yystate = state_peek(0);     //get new state
    val_drop(yym);               //corresponding value drop
    yym = yylhs[yyn];            //select next TERMINAL(on lhs)
    if (yystate == 0 && yym == 0)//done? 'rest' state and at first TERMINAL
      {
      //if (yydebug) debug("After reduction, shifting from state 0 to state "+YYFINAL+"");
      yystate = YYFINAL;         //explicitly say we're done
      state_push(YYFINAL);       //and save it
      val_push(yyval);           //also save the semantic value of parsing
      if (yychar < 0)            //we want another character?
        {
        yychar = yylex();        //get next character
        //if (yychar<0) yychar=0;  //clean, if necessary
        //if (yydebug)
          //yylexdebug(yystate,yychar);
        }
      if (yychar == 0)          //Good exit (if lex returns 0 ;-)
         break;                 //quit the loop--all DONE
      }//if yystate
    else                        //else not done yet
      {                         //get next state and push, for next yydefred[]
      yyn = yygindex[yym];      //find out where to go
      if ((yyn != 0) && (yyn += yystate) >= 0 &&
            yyn <= YYTABLESIZE && yycheck[yyn] == yystate)
        yystate = yytable[yyn]; //get new state
      else
        yystate = yydgoto[yym]; //else go to new defred
      //if (yydebug) debug("after reduction, shifting from state "+state_peek(0)+" to state "+yystate+"");
      state_push(yystate);     //going again, so push state & val...
      val_push(yyval);         //for next action
      }
    }//main loop
  return 0;//yyaccept!!
}
//## end of method parse() ######################################



//## run() --- for Thread #######################################
//## The -Jnorun option was used ##
//## end of method run() ########################################



//## Constructors ###############################################
//## The -Jnoconstruct option was used ##
//###############################################################



}
//################### END OF CLASS ##############################
