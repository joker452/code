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
public final static short UMINUS=288;
public final static short EMPTY=289;
public final static short YYERRCODE=256;
final static short yylhs[] = {                           -1,
    0,    1,    1,    3,    4,    5,    5,    5,    5,    5,
    5,    2,    2,    6,    6,    7,    7,    7,    9,    9,
   10,   10,    8,    8,   11,   12,   12,   13,   13,   13,
   13,   13,   13,   13,   13,   13,   13,   13,   21,   22,
   22,   24,   24,   25,   14,   14,   14,   28,   28,   26,
   26,   27,   23,   23,   23,   23,   23,   23,   23,   23,
   23,   23,   23,   23,   23,   23,   23,   23,   23,   23,
   23,   23,   23,   23,   23,   23,   23,   23,   30,   30,
   29,   29,   31,   31,   16,   17,   20,   15,   32,   32,
   18,   18,   19,
};
final static short yylen[] = {                            2,
    1,    2,    1,    2,    2,    1,    1,    1,    1,    2,
    3,    7,    6,    2,    0,    2,    2,    0,    1,    0,
    3,    1,    7,    6,    3,    2,    0,    1,    2,    1,
    1,    1,    2,    2,    2,    2,    1,    1,    6,    5,
    3,    3,    0,    3,    3,    1,    0,    2,    0,    2,
    4,    5,    1,    1,    1,    3,    3,    3,    3,    3,
    3,    3,    3,    3,    3,    3,    3,    3,    3,    2,
    2,    3,    3,    1,    4,    5,    6,    5,    1,    1,
    1,    0,    3,    1,    5,    9,    1,    6,    2,    0,
    2,    1,    4,
};
final static short yydefred[] = {                         0,
    0,    0,    0,    0,    3,    0,    0,    2,    0,    0,
    0,   14,   18,    0,    0,   18,    7,    8,    6,    9,
    0,    0,   13,   16,    0,    0,   17,    0,   10,    0,
    4,    0,    0,   12,    0,    0,   11,    0,   22,    0,
    0,    0,    0,    5,    0,    0,    0,   27,   24,   21,
   23,    0,   80,   74,    0,    0,    0,    0,   87,    0,
    0,    0,    0,   79,    0,    0,    0,    0,   25,    0,
   28,   38,   26,    0,   30,   31,   32,    0,    0,    0,
    0,   37,    0,    0,    0,    0,   55,    0,    0,    0,
    0,    0,   53,   54,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,   29,   33,   34,   35,   36,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,   48,    0,    0,    0,    0,    0,    0,
   41,    0,    0,    0,    0,    0,   72,   73,    0,    0,
   69,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,   75,    0,    0,   93,    0,    0,    0,
   51,    0,    0,   85,    0,    0,    0,   40,   42,   76,
    0,    0,   78,    0,   52,    0,    0,   88,   44,   77,
   39,    0,   89,    0,   86,
};
final static short yydgoto[] = {                          3,
    4,    5,   71,   25,   40,   10,   15,   27,   41,   42,
   72,   52,   73,   74,   75,   76,   77,   78,   79,   80,
   81,   82,   83,  132,  163,   93,   94,   86,  172,   87,
  136,  188,
};
final static short yysindex[] = {                      -240,
 -252, -224,    0, -240,    0, -215, -213,    0, -207,  -53,
 -215,    0,    0,  -52,  -76,    0,    0,    0,    0,    0,
 -201,  -83,    0,    0,   34,  -88,    0,  -49,    0,  -87,
    0,   60,    9,    0,   74,  -83,    0,  -83,    0,  -85,
   89,   71,   90,    0,   13,  -83,   13,    0,    0,    0,
    0,   -2,    0,    0,   92,   98,  -22,  641,    0, -172,
  100,  101,  117,    0,  123,  641,  641,  409,    0,  125,
    0,    0,    0,   83,    0,    0,    0,  107,  108,  113,
  121,    0,  543,  126,    0, -125,    0,  641,  641,  641,
   61,  543,    0,    0,  150,  103,  641,  152,  160,  641,
  -26,  -26,  -71,  383,  -70,    0,    0,    0,    0,    0,
  641,  641,  641,  641,  641,  641,  641,  641,  641,  641,
  641,  641,  641,    0,  641,  641,  167,  410,  157,  421,
    0,  641,  179,  613,  543,  -12,    0,    0,  279,  180,
    0,  178,  702,  585,    8,    8,  -32,  -32,   15,   15,
  -26,  -26,  -26,    8,    8,  442,  543,  641,   27,  641,
   27,  453, -108,    0,  477,  641,    0,  -41,  641,  641,
    0,  182,  200,    0,  505,  -23,   27,    0,    0,    0,
  543,  205,    0,  532,    0,  641,   27,    0,    0,    0,
    0,  210,    0,   27,    0,
};
final static short yyrindex[] = {                         0,
    0,    0,    0,  261,    0,  143,    0,    0,    0,    0,
  143,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,  215,    0,    0,    0,  235,    0,  235,    0,    0,
    0,  238,    0,    0,    0,    0,    0,    0,    0,    0,
    0,  -58,    0,    0,    0,    0,    0,  -57,    0,    0,
    0,    0,    0,    0,    0,    4,    4,    4,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,  564,  359,    0,    0,    4,  -58,    4,
  501,  222,    0,    0,    0,    0,    4,    0,    0,    4,
   66,   75,    0,    0,    0,    0,    0,    0,    0,    0,
    4,    4,    4,    4,    4,    4,    4,    4,    4,    4,
    4,    4,    4,    0,    4,    4,   36,    0,    0,    0,
    0,    4,    0,    4,   -5,    0,    0,    0,    0,    0,
    0,    0,  450,  -25,  434,  712,  839,  850,  792,  814,
  102,  111,  155,  760,  837,    0,  -18,   -1,  -58,    4,
  -58,    0,    0,    0,    0,    4,    0,    0,    4,    4,
    0,    0,  241,    0,    0,  -33,  -58,    0,    0,    0,
    3,    0,    0,    0,    0,    1,  -58,    0,    0,    0,
    0,    0,    0,  -58,    0,
};
final static short yygindex[] = {                         0,
    0,  286,   -6,  -11,  312,  284,  287,    0,  266,    0,
   19,    0, -103,  -81,    0,    0,    0,    0,    0,    0,
    0,    0,  891,    0,    0,  740,  776,    0,    0,    0,
  147,    0,
};
final static int YYTABLESIZE=1128;
static short yytable[];
static { yytable();}
static void yytable(){
yytable = new short[]{                         90,
   47,   92,   33,   33,  121,   33,   90,  129,   24,  119,
  117,   90,  118,  124,  120,   68,  178,   90,   68,  124,
    1,   24,   45,    6,   39,   90,   39,  123,  167,  122,
   67,  166,   68,   68,   50,   84,    7,   68,   84,   82,
   45,   47,   66,   83,  121,    2,   83,    9,   23,  119,
  117,  121,  118,  124,  120,  174,  119,  176,  125,   67,
  124,  120,   11,   49,  125,   51,   68,   68,   12,   13,
   16,   66,   50,  189,   29,   34,   50,   50,   50,   50,
   50,   50,   50,  193,   17,   18,   19,   20,   21,   90,
  195,   90,   31,   50,   50,   50,   50,   50,  125,   36,
   91,   37,   70,   95,  192,  125,   70,   70,   70,   70,
   70,   71,   70,   38,   46,   71,   71,   71,   71,   71,
   48,   71,   69,   70,   70,   70,   50,   70,   50,   45,
   47,   88,   71,   71,   71,   48,   71,   89,   58,   97,
   98,  106,   58,   58,   58,   58,   58,   59,   58,   48,
  127,   59,   59,   59,   59,   59,   99,   59,   70,   58,
   58,   58,  100,   58,  105,  107,  108,   71,   59,   59,
   59,  109,   59,   17,   18,   19,   20,   21,  179,  110,
   17,   18,   19,   20,   21,  131,  126,   32,   35,  133,
   44,   60,  137,  134,   58,   60,   60,   60,   60,   60,
  138,   60,   22,   59,  140,  142,  158,   17,   18,   19,
   20,   21,   60,   60,   60,  160,   60,   49,   49,  164,
  169,  170,  185,   90,   90,   90,   90,   90,   90,   22,
   90,   90,   90,   90,  182,   90,   90,   90,   90,   90,
   90,   90,   90,  166,  187,  190,   90,   60,  113,  114,
  194,   90,   68,   90,   17,   18,   19,   20,   21,   53,
    1,   54,   55,   56,   57,   15,   58,   59,   60,   61,
   62,   63,   64,    5,   49,   20,   49,   65,   19,   49,
   91,   81,   70,   17,   18,   19,   20,   21,   53,    8,
   54,   55,   56,   57,   14,   58,   59,   60,   61,   62,
   63,   64,   28,   43,  173,    0,   65,    0,    0,    0,
    0,   70,   50,   50,    0,  121,   50,   50,   50,   50,
  119,  117,  168,  118,  124,  120,   26,    0,    0,    0,
    0,    0,    0,   30,    0,    0,    0,    0,  123,   26,
  122,    0,   70,   70,    0,    0,   70,   70,   70,   70,
    0,   71,   71,    0,    0,   71,   71,   71,   71,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,  125,
    0,   96,    0,    0,    0,    0,    0,    0,   58,   58,
    0,    0,   58,   58,   58,   58,    0,   59,   59,    0,
    0,   59,   59,   59,   59,   54,    0,    0,    0,   46,
   54,   54,    0,   54,   54,   54,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,   46,   54,  121,
   54,    0,    0,  141,  119,  117,    0,  118,  124,  120,
    0,   60,   60,    0,    0,   60,   60,   60,   60,    0,
    0,   67,  123,    0,  122,    0,  121,    0,   68,   54,
  159,  119,  117,   66,  118,  124,  120,  121,    0,    0,
    0,  161,  119,  117,    0,  118,  124,  120,    0,  123,
    0,  122,    0,  125,   65,    0,    0,   65,  121,    0,
  123,    0,  122,  119,  117,    0,  118,  124,  120,  121,
   67,   65,   65,   67,  119,  117,    0,  118,  124,  120,
  125,  123,    0,  122,    0,    0,    0,   67,   67,    0,
  177,  125,  123,  121,  122,    0,    0,    0,  119,  117,
    0,  118,  124,  120,    0,    0,   65,    0,    0,    0,
    0,    0,  125,   43,  171,    0,  123,    0,  122,    0,
   43,  121,   67,  125,    0,   43,  119,  117,    0,  118,
  124,  120,    0,    0,    0,  111,  112,    0,    0,  113,
  114,  115,  116,  186,  123,    0,  122,  125,  121,  180,
    0,    0,  191,  119,  117,    0,  118,  124,  120,  121,
    0,    0,    0,    0,  119,  117,    0,  118,  124,  120,
    0,  123,    0,  122,    0,  125,    0,    0,    0,    0,
   53,    0,  123,    0,  122,   53,   53,    0,   53,   53,
   53,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,  121,  125,   53,    0,   53,  119,  117,    0,  118,
  124,  120,    0,  125,    0,   54,   54,    0,    0,   54,
   54,   54,   54,    0,  123,   67,  122,    0,    0,    0,
    0,    0,   68,    0,   53,    0,    0,   66,    0,  111,
  112,    0,    0,  113,  114,  115,  116,    0,    0,  103,
   53,    0,   54,   67,    0,  125,    0,    0,    0,   60,
   68,   62,   63,   64,    0,   66,  111,  112,   65,    0,
  113,  114,  115,  116,    0,    0,    0,  111,  112,    0,
    0,  113,  114,  115,  116,   37,    0,    0,    0,    0,
   65,   65,    0,    0,    0,    0,   65,   65,  111,  112,
    0,    0,  113,  114,  115,  116,   67,   67,    0,  111,
  112,    0,    0,  113,  114,  115,  116,    0,  121,    0,
    0,    0,    0,  119,  117,    0,  118,  124,  120,    0,
    0,    0,   66,  111,  112,   66,    0,  113,  114,  115,
  116,  123,   43,  122,   43,    0,    0,    0,    0,   66,
   66,   43,    0,   43,   43,   43,   43,    0,    0,    0,
   43,  111,  112,    0,    0,  113,  114,  115,  116,    0,
    0,   84,  125,    0,    0,    0,    0,    0,    0,    0,
   64,    0,    0,   64,   66,    0,    0,    0,  111,  112,
    0,    0,  113,  114,  115,  116,    0,   64,   64,  111,
  112,    0,    0,  113,  114,  115,  116,   85,   84,    0,
    0,    0,   56,    0,   56,   56,   56,    0,    0,    0,
   53,   53,    0,    0,   53,   53,   53,   53,    0,   56,
   56,   56,   64,   56,   57,    0,   57,   57,   57,    0,
    0,  111,    0,    0,   85,  113,  114,  115,  116,    0,
    0,   57,   57,   57,   53,   57,   54,   63,    0,   61,
   63,    0,   61,   60,   56,   62,   63,   64,    0,    0,
   62,    0,   65,   62,   63,   63,   61,   61,   84,    0,
   84,    0,   53,    0,   54,    0,   57,   62,   62,    0,
    0,   60,    0,   62,   63,   64,   84,    0,    0,    0,
   65,    0,    0,    0,    0,   84,   84,    0,    0,   63,
    0,   61,    0,   84,   85,    0,   85,    0,    0,    0,
    0,    0,   62,    0,    0,    0,    0,    0,   92,    0,
    0,    0,   85,    0,    0,    0,  101,  102,  104,    0,
    0,   85,   85,    0,    0,    0,    0,    0,    0,   85,
    0,    0,    0,    0,    0,    0,    0,    0,  128,    0,
  130,    0,  113,  114,  115,  116,    0,  135,   66,   66,
  139,    0,    0,    0,   66,   66,    0,    0,    0,    0,
    0,  143,  144,  145,  146,  147,  148,  149,  150,  151,
  152,  153,  154,  155,    0,  156,  157,    0,    0,    0,
    0,    0,  162,    0,  165,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,   64,   64,    0,    0,
    0,    0,   64,   64,    0,    0,    0,    0,  135,    0,
  175,    0,    0,    0,    0,    0,  181,    0,    0,  183,
  184,    0,    0,    0,    0,    0,    0,    0,   56,   56,
    0,    0,   56,   56,   56,   56,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
   57,   57,    0,    0,   57,   57,   57,   57,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,   63,   63,   61,   61,    0,    0,   63,
   63,    0,    0,    0,    0,    0,   62,   62,
};
}
static short yycheck[];
static { yycheck(); }
static void yycheck() {
yycheck = new short[] {                         33,
   59,   59,   91,   91,   37,   91,   40,   89,   15,   42,
   43,   45,   45,   46,   47,   41,  125,   40,   44,   46,
  261,   28,   41,  276,   36,   59,   38,   60,   41,   62,
   33,   44,   58,   59,   46,   41,  261,   40,   44,   41,
   59,   41,   45,   41,   37,  286,   44,  263,  125,   42,
   43,   37,   45,   46,   47,  159,   42,  161,   91,   33,
   46,   47,  276,   45,   91,   47,   40,   93,  276,  123,
  123,   45,   37,  177,  276,  125,   41,   42,   43,   44,
   45,   46,   47,  187,  257,  258,  259,  260,  261,  123,
  194,  125,   59,   58,   59,   60,   61,   62,   91,   40,
  123,   93,   37,  276,  186,   91,   41,   42,   43,   44,
   45,   37,   47,   40,   44,   41,   42,   43,   44,   45,
  123,   47,  125,   58,   59,   60,   91,   62,   93,   41,
   41,   40,   58,   59,   60,  123,   62,   40,   37,   40,
   40,   59,   41,   42,   43,   44,   45,   37,   47,  123,
  276,   41,   42,   43,   44,   45,   40,   47,   93,   58,
   59,   60,   40,   62,   40,   59,   59,   93,   58,   59,
   60,   59,   62,  257,  258,  259,  260,  261,  287,   59,
  257,  258,  259,  260,  261,  125,   61,  276,  276,   40,
  276,   37,   41,   91,   93,   41,   42,   43,   44,   45,
   41,   47,  279,   93,  276,  276,   40,  257,  258,  259,
  260,  261,   58,   59,   60,   59,   62,  276,  276,   41,
   41,   44,   41,  257,  258,  259,  260,  261,  262,  279,
  264,  265,  266,  267,  276,  269,  270,  271,  272,  273,
  274,  275,  276,   44,  268,   41,  280,   93,  281,  282,
   41,  285,  278,  287,  257,  258,  259,  260,  261,  262,
    0,  264,  265,  266,  267,  123,  269,  270,  271,  272,
  273,  274,  275,   59,  276,   41,  276,  280,   41,  276,
   59,   41,  285,  257,  258,  259,  260,  261,  262,    4,
  264,  265,  266,  267,   11,  269,  270,  271,  272,  273,
  274,  275,   16,   38,  158,   -1,  280,   -1,   -1,   -1,
   -1,  285,  277,  278,   -1,   37,  281,  282,  283,  284,
   42,   43,   44,   45,   46,   47,   15,   -1,   -1,   -1,
   -1,   -1,   -1,   22,   -1,   -1,   -1,   -1,   60,   28,
   62,   -1,  277,  278,   -1,   -1,  281,  282,  283,  284,
   -1,  277,  278,   -1,   -1,  281,  282,  283,  284,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   91,
   -1,   60,   -1,   -1,   -1,   -1,   -1,   -1,  277,  278,
   -1,   -1,  281,  282,  283,  284,   -1,  277,  278,   -1,
   -1,  281,  282,  283,  284,   37,   -1,   -1,   -1,   41,
   42,   43,   -1,   45,   46,   47,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   59,   60,   37,
   62,   -1,   -1,   41,   42,   43,   -1,   45,   46,   47,
   -1,  277,  278,   -1,   -1,  281,  282,  283,  284,   -1,
   -1,   33,   60,   -1,   62,   -1,   37,   -1,   40,   91,
   41,   42,   43,   45,   45,   46,   47,   37,   -1,   -1,
   -1,   41,   42,   43,   -1,   45,   46,   47,   -1,   60,
   -1,   62,   -1,   91,   41,   -1,   -1,   44,   37,   -1,
   60,   -1,   62,   42,   43,   -1,   45,   46,   47,   37,
   41,   58,   59,   44,   42,   43,   -1,   45,   46,   47,
   91,   60,   -1,   62,   -1,   -1,   -1,   58,   59,   -1,
   58,   91,   60,   37,   62,   -1,   -1,   -1,   42,   43,
   -1,   45,   46,   47,   -1,   -1,   93,   -1,   -1,   -1,
   -1,   -1,   91,   33,   93,   -1,   60,   -1,   62,   -1,
   40,   37,   93,   91,   -1,   45,   42,   43,   -1,   45,
   46,   47,   -1,   -1,   -1,  277,  278,   -1,   -1,  281,
  282,  283,  284,   59,   60,   -1,   62,   91,   37,   93,
   -1,   -1,   41,   42,   43,   -1,   45,   46,   47,   37,
   -1,   -1,   -1,   -1,   42,   43,   -1,   45,   46,   47,
   -1,   60,   -1,   62,   -1,   91,   -1,   -1,   -1,   -1,
   37,   -1,   60,   -1,   62,   42,   43,   -1,   45,   46,
   47,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   37,   91,   60,   -1,   62,   42,   43,   -1,   45,
   46,   47,   -1,   91,   -1,  277,  278,   -1,   -1,  281,
  282,  283,  284,   -1,   60,   33,   62,   -1,   -1,   -1,
   -1,   -1,   40,   -1,   91,   -1,   -1,   45,   -1,  277,
  278,   -1,   -1,  281,  282,  283,  284,   -1,   -1,  261,
  262,   -1,  264,   33,   -1,   91,   -1,   -1,   -1,  271,
   40,  273,  274,  275,   -1,   45,  277,  278,  280,   -1,
  281,  282,  283,  284,   -1,   -1,   -1,  277,  278,   -1,
   -1,  281,  282,  283,  284,   93,   -1,   -1,   -1,   -1,
  277,  278,   -1,   -1,   -1,   -1,  283,  284,  277,  278,
   -1,   -1,  281,  282,  283,  284,  277,  278,   -1,  277,
  278,   -1,   -1,  281,  282,  283,  284,   -1,   37,   -1,
   -1,   -1,   -1,   42,   43,   -1,   45,   46,   47,   -1,
   -1,   -1,   41,  277,  278,   44,   -1,  281,  282,  283,
  284,   60,  262,   62,  264,   -1,   -1,   -1,   -1,   58,
   59,  271,   -1,  273,  274,  275,  276,   -1,   -1,   -1,
  280,  277,  278,   -1,   -1,  281,  282,  283,  284,   -1,
   -1,   52,   91,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   41,   -1,   -1,   44,   93,   -1,   -1,   -1,  277,  278,
   -1,   -1,  281,  282,  283,  284,   -1,   58,   59,  277,
  278,   -1,   -1,  281,  282,  283,  284,   52,   89,   -1,
   -1,   -1,   41,   -1,   43,   44,   45,   -1,   -1,   -1,
  277,  278,   -1,   -1,  281,  282,  283,  284,   -1,   58,
   59,   60,   93,   62,   41,   -1,   43,   44,   45,   -1,
   -1,  277,   -1,   -1,   89,  281,  282,  283,  284,   -1,
   -1,   58,   59,   60,  262,   62,  264,   41,   -1,   41,
   44,   -1,   44,  271,   93,  273,  274,  275,   -1,   -1,
   41,   -1,  280,   44,   58,   59,   58,   59,  159,   -1,
  161,   -1,  262,   -1,  264,   -1,   93,   58,   59,   -1,
   -1,  271,   -1,  273,  274,  275,  177,   -1,   -1,   -1,
  280,   -1,   -1,   -1,   -1,  186,  187,   -1,   -1,   93,
   -1,   93,   -1,  194,  159,   -1,  161,   -1,   -1,   -1,
   -1,   -1,   93,   -1,   -1,   -1,   -1,   -1,   58,   -1,
   -1,   -1,  177,   -1,   -1,   -1,   66,   67,   68,   -1,
   -1,  186,  187,   -1,   -1,   -1,   -1,   -1,   -1,  194,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   88,   -1,
   90,   -1,  281,  282,  283,  284,   -1,   97,  277,  278,
  100,   -1,   -1,   -1,  283,  284,   -1,   -1,   -1,   -1,
   -1,  111,  112,  113,  114,  115,  116,  117,  118,  119,
  120,  121,  122,  123,   -1,  125,  126,   -1,   -1,   -1,
   -1,   -1,  132,   -1,  134,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,  277,  278,   -1,   -1,
   -1,   -1,  283,  284,   -1,   -1,   -1,   -1,  158,   -1,
  160,   -1,   -1,   -1,   -1,   -1,  166,   -1,   -1,  169,
  170,   -1,   -1,   -1,   -1,   -1,   -1,   -1,  277,  278,
   -1,   -1,  281,  282,  283,  284,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
  277,  278,   -1,   -1,  281,  282,  283,  284,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,  277,  278,  277,  278,   -1,   -1,  283,
  284,   -1,   -1,   -1,   -1,   -1,  277,  278,
};
}
final static short YYFINAL=3;
final static short YYMAXTOKEN=289;
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
"SEALED","SEP","UMINUS","EMPTY",
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

//#line 463 "Parser.y"
    
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
//#line 624 "Parser.java"
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
//#line 52 "Parser.y"
{
						tree = new Tree.TopLevel(val_peek(0).clist, val_peek(0).loc);
					}
break;
case 2:
//#line 58 "Parser.y"
{
						yyval.clist.add(val_peek(0).cdef);
					}
break;
case 3:
//#line 62 "Parser.y"
{
                		yyval.clist = new ArrayList<Tree.ClassDef>();
                		yyval.clist.add(val_peek(0).cdef);
                	}
break;
case 5:
//#line 72 "Parser.y"
{
						yyval.vdef = new Tree.VarDef(val_peek(0).ident, val_peek(1).type, val_peek(0).loc);
					}
break;
case 6:
//#line 78 "Parser.y"
{
						yyval.type = new Tree.TypeIdent(Tree.INT, val_peek(0).loc);
					}
break;
case 7:
//#line 82 "Parser.y"
{
                		yyval.type = new Tree.TypeIdent(Tree.VOID, val_peek(0).loc);
                	}
break;
case 8:
//#line 86 "Parser.y"
{
                		yyval.type = new Tree.TypeIdent(Tree.BOOL, val_peek(0).loc);
                	}
break;
case 9:
//#line 90 "Parser.y"
{
                		yyval.type = new Tree.TypeIdent(Tree.STRING, val_peek(0).loc);
                	}
break;
case 10:
//#line 94 "Parser.y"
{
                		yyval.type = new Tree.TypeClass(val_peek(0).ident, val_peek(1).loc);
                	}
break;
case 11:
//#line 98 "Parser.y"
{
                		yyval.type = new Tree.TypeArray(val_peek(2).type, val_peek(2).loc);
                	}
break;
case 12:
//#line 104 "Parser.y"
{
						yyval.cdef = new Tree.ClassDef(true, val_peek(4).ident, val_peek(3).ident, val_peek(1).flist, val_peek(5).loc);
					}
break;
case 13:
//#line 108 "Parser.y"
{
						yyval.cdef = new Tree.ClassDef(false, val_peek(4).ident, val_peek(3).ident, val_peek(1).flist, val_peek(5).loc);
					}
break;
case 14:
//#line 114 "Parser.y"
{
						yyval.ident = val_peek(0).ident;
					}
break;
case 15:
//#line 118 "Parser.y"
{
                		yyval = new SemValue();
                	}
break;
case 16:
//#line 124 "Parser.y"
{
						yyval.flist.add(val_peek(0).vdef);
					}
break;
case 17:
//#line 128 "Parser.y"
{
						yyval.flist.add(val_peek(0).fdef);
					}
break;
case 18:
//#line 132 "Parser.y"
{
                		yyval = new SemValue();
                		yyval.flist = new ArrayList<Tree>();
                	}
break;
case 20:
//#line 140 "Parser.y"
{
                		yyval = new SemValue();
                		yyval.vlist = new ArrayList<Tree.VarDef>(); 
                	}
break;
case 21:
//#line 147 "Parser.y"
{
						yyval.vlist.add(val_peek(0).vdef);
					}
break;
case 22:
//#line 151 "Parser.y"
{
                		yyval.vlist = new ArrayList<Tree.VarDef>();
						yyval.vlist.add(val_peek(0).vdef);
                	}
break;
case 23:
//#line 158 "Parser.y"
{
						yyval.fdef = new MethodDef(true, val_peek(4).ident, val_peek(5).type, val_peek(2).vlist, (Block) val_peek(0).stmt, val_peek(4).loc);
					}
break;
case 24:
//#line 162 "Parser.y"
{
						yyval.fdef = new MethodDef(false, val_peek(4).ident, val_peek(5).type, val_peek(2).vlist, (Block) val_peek(0).stmt, val_peek(4).loc);
					}
break;
case 25:
//#line 168 "Parser.y"
{
						yyval.stmt = new Block(val_peek(1).slist, val_peek(2).loc);
					}
break;
case 26:
//#line 174 "Parser.y"
{
						yyval.slist.add(val_peek(0).stmt);
					}
break;
case 27:
//#line 178 "Parser.y"
{
                		yyval = new SemValue();
                		yyval.slist = new ArrayList<Tree>();
                	}
break;
case 28:
//#line 185 "Parser.y"
{
						yyval.stmt = val_peek(0).vdef;
					}
break;
case 29:
//#line 190 "Parser.y"
{
                		if (yyval.stmt == null) {
                			yyval.stmt = new Tree.Skip(val_peek(0).loc);
                		}
                	}
break;
case 39:
//#line 207 "Parser.y"
{
						yyval.stmt = new Tree.Scopy(val_peek(1).expr, val_peek(3).ident, val_peek(5).loc);
					}
break;
case 40:
//#line 213 "Parser.y"
{
                        val_peek(2).slist.add(val_peek(1).stmt);
                        yyval.stmt = new Tree.GuardStmt(val_peek(2).slist, val_peek(4).loc);
                    }
break;
case 41:
//#line 218 "Parser.y"
{
				        yyval.stmt = new Tree.GuardStmt(null, val_peek(2).loc);
				    }
break;
case 42:
//#line 224 "Parser.y"
{
                       yyval.slist.add(val_peek(1).stmt);
                    }
break;
case 43:
//#line 228 "Parser.y"
{
				    	yyval = new SemValue();
				    	yyval.slist = new ArrayList<Tree>();
				    }
break;
case 44:
//#line 235 "Parser.y"
{
						yyval.stmt = new Tree.Guard(val_peek(2).expr, val_peek(0).stmt, val_peek(2).loc);
					}
break;
case 45:
//#line 241 "Parser.y"
{
						yyval.stmt = new Tree.Assign(val_peek(2).lvalue, val_peek(0).expr, val_peek(1).loc);
					}
break;
case 46:
//#line 245 "Parser.y"
{
                		yyval.stmt = new Tree.Exec(val_peek(0).expr, val_peek(0).loc);
                	}
break;
case 47:
//#line 249 "Parser.y"
{
                		yyval = new SemValue();
                	}
break;
case 49:
//#line 256 "Parser.y"
{
                		yyval = new SemValue();
                	}
break;
case 50:
//#line 262 "Parser.y"
{
						yyval.lvalue = new Tree.Ident(val_peek(1).expr, val_peek(0).ident, val_peek(0).loc);
						if (val_peek(1).loc == null) {
							yyval.loc = val_peek(0).loc;
						}
					}
break;
case 51:
//#line 269 "Parser.y"
{
                		yyval.lvalue = new Tree.Indexed(val_peek(3).expr, val_peek(1).expr, val_peek(3).loc);
                	}
break;
case 52:
//#line 275 "Parser.y"
{
						yyval.expr = new Tree.CallExpr(val_peek(4).expr, val_peek(3).ident, val_peek(1).elist, val_peek(3).loc);
						if (val_peek(4).loc == null) {
							yyval.loc = val_peek(3).loc;
						}
					}
break;
case 53:
//#line 284 "Parser.y"
{
						yyval.expr = val_peek(0).lvalue;
					}
break;
case 56:
//#line 290 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.PLUS, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 57:
//#line 294 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.MINUS, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 58:
//#line 298 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.MUL, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 59:
//#line 302 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.DIV, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 60:
//#line 306 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.MOD, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 61:
//#line 310 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.EQ, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 62:
//#line 314 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.NE, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 63:
//#line 318 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.LT, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 64:
//#line 322 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.GT, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 65:
//#line 326 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.LE, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 66:
//#line 330 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.GE, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 67:
//#line 334 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.AND, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 68:
//#line 338 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.OR, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 69:
//#line 342 "Parser.y"
{
                		yyval = val_peek(1);
                	}
break;
case 70:
//#line 346 "Parser.y"
{
                		yyval.expr = new Tree.Unary(Tree.NEG, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 71:
//#line 350 "Parser.y"
{
                		yyval.expr = new Tree.Unary(Tree.NOT, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 72:
//#line 354 "Parser.y"
{
                		yyval.expr = new Tree.ReadIntExpr(val_peek(2).loc);
                	}
break;
case 73:
//#line 358 "Parser.y"
{
                		yyval.expr = new Tree.ReadLineExpr(val_peek(2).loc);
                	}
break;
case 74:
//#line 362 "Parser.y"
{
                		yyval.expr = new Tree.ThisExpr(val_peek(0).loc);
                	}
break;
case 75:
//#line 366 "Parser.y"
{
                		yyval.expr = new Tree.NewClass(val_peek(2).ident, val_peek(3).loc);
                	}
break;
case 76:
//#line 370 "Parser.y"
{
                		yyval.expr = new Tree.NewArray(val_peek(3).type, val_peek(1).expr, val_peek(4).loc);
                	}
break;
case 77:
//#line 374 "Parser.y"
{
                		yyval.expr = new Tree.TypeTest(val_peek(3).expr, val_peek(1).ident, val_peek(5).loc);
                	}
break;
case 78:
//#line 378 "Parser.y"
{
                		yyval.expr = new Tree.TypeCast(val_peek(2).ident, val_peek(0).expr, val_peek(0).loc);
                	}
break;
case 79:
//#line 384 "Parser.y"
{
						yyval.expr = new Tree.Literal(val_peek(0).typeTag, val_peek(0).literal, val_peek(0).loc);
					}
break;
case 80:
//#line 388 "Parser.y"
{
						yyval.expr = new Null(val_peek(0).loc);
					}
break;
case 82:
//#line 395 "Parser.y"
{
                		yyval = new SemValue();
                		yyval.elist = new ArrayList<Tree.Expr>();
                	}
break;
case 83:
//#line 402 "Parser.y"
{
						yyval.elist.add(val_peek(0).expr);
					}
break;
case 84:
//#line 406 "Parser.y"
{
                		yyval.elist = new ArrayList<Tree.Expr>();
						yyval.elist.add(val_peek(0).expr);
                	}
break;
case 85:
//#line 413 "Parser.y"
{
						yyval.stmt = new Tree.WhileLoop(val_peek(2).expr, val_peek(0).stmt, val_peek(4).loc);
					}
break;
case 86:
//#line 419 "Parser.y"
{
						yyval.stmt = new Tree.ForLoop(val_peek(6).stmt, val_peek(4).expr, val_peek(2).stmt, val_peek(0).stmt, val_peek(8).loc);
					}
break;
case 87:
//#line 425 "Parser.y"
{
						yyval.stmt = new Tree.Break(val_peek(0).loc);
					}
break;
case 88:
//#line 431 "Parser.y"
{
						yyval.stmt = new Tree.If(val_peek(3).expr, val_peek(1).stmt, val_peek(0).stmt, val_peek(5).loc);
					}
break;
case 89:
//#line 437 "Parser.y"
{
						yyval.stmt = val_peek(0).stmt;
					}
break;
case 90:
//#line 441 "Parser.y"
{
						yyval = new SemValue();
					}
break;
case 91:
//#line 447 "Parser.y"
{
						yyval.stmt = new Tree.Return(val_peek(0).expr, val_peek(1).loc);
					}
break;
case 92:
//#line 451 "Parser.y"
{
                		yyval.stmt = new Tree.Return(null, val_peek(0).loc);
                	}
break;
case 93:
//#line 457 "Parser.y"
{
						yyval.stmt = new Print(val_peek(1).elist, val_peek(3).loc);
					}
break;
//#line 1255 "Parser.java"
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
