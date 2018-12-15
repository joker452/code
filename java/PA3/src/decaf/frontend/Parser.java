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
public final static short UMINUS=287;
public final static short EMPTY=288;
public final static short YYERRCODE=256;
final static short yylhs[] = {                           -1,
    0,    1,    1,    3,    4,    5,    5,    5,    5,    5,
    5,    2,    2,    6,    6,    7,    7,    7,    9,    9,
   10,   10,    8,    8,   11,   12,   12,   13,   13,   13,
   13,   13,   13,   13,   13,   13,   13,   21,   14,   14,
   14,   25,   25,   23,   23,   24,   22,   22,   22,   22,
   22,   22,   22,   22,   22,   22,   22,   22,   22,   22,
   22,   22,   22,   22,   22,   22,   22,   22,   22,   22,
   22,   22,   27,   27,   26,   26,   28,   28,   16,   17,
   20,   15,   29,   29,   18,   18,   19,
};
final static short yylen[] = {                            2,
    1,    2,    1,    2,    2,    1,    1,    1,    1,    2,
    3,    7,    6,    2,    0,    2,    2,    0,    1,    0,
    3,    1,    7,    6,    3,    2,    0,    1,    2,    1,
    1,    1,    2,    2,    2,    2,    1,    6,    3,    1,
    0,    2,    0,    2,    4,    5,    1,    1,    1,    3,
    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,
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
   23,    0,   74,   68,    0,    0,    0,    0,   81,    0,
    0,    0,    0,   73,    0,    0,    0,    0,   25,    0,
   28,   37,   26,    0,   30,   31,   32,    0,    0,    0,
    0,    0,    0,    0,    0,   49,    0,    0,    0,    0,
   47,   48,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,   29,   33,   34,   35,   36,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,   42,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,   66,   67,    0,    0,   63,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,   69,    0,    0,
   87,    0,    0,    0,   45,    0,    0,   79,    0,    0,
   70,    0,    0,   72,    0,   46,    0,    0,   82,   71,
   38,    0,   83,    0,   80,
};
final static short yydgoto[] = {                          3,
    4,    5,   71,   25,   40,   10,   15,   27,   41,   42,
   72,   52,   73,   74,   75,   76,   77,   78,   79,   80,
   81,   82,   91,   92,   85,  166,   86,  132,  179,
};
final static short yysindex[] = {                      -253,
 -249, -227,    0, -253,    0, -228, -234,    0, -230,  -76,
 -228,    0,    0,  -68,  -80,    0,    0,    0,    0,    0,
 -216,  -50,    0,    0,    2,  -89,    0,  328,    0,  -87,
    0,   27,  -11,    0,   49,  -50,    0,  -50,    0,  -85,
   58,   22,   63,    0,  -15,  -50,  -15,    0,    0,    0,
    0,   -4,    0,    0,   74,   78,   80,  526,    0, -174,
   82,   89,   90,    0,   99,  526,  526,  501,    0,  106,
    0,    0,    0,   81,    0,    0,    0,   93,   97,  102,
  105,  535,   88,    0, -118,    0,  526,  526,  526,  535,
    0,    0,  134,   75,  526,  135,  141,  526,  -29,  -29,
 -101,  351,  -93,    0,    0,    0,    0,    0,  526,  526,
  526,  526,  526,  526,  526,  526,  526,  526,  526,  526,
  526,    0,  526,  526,  152,  378,  142,  402,  161,  606,
  535,   47,    0,    0,  413,  162,    0,  160,  696,  578,
    6,    6,  -32,  -32,   26,   26,  -29,  -29,  -29,    6,
    6,  434,  535,  526,   25,  526,   25,    0,  445,  526,
    0,  -71,  526,  526,    0,  173,  172,    0,  469,  -51,
    0,  535,  180,    0,  502,    0,  526,   25,    0,    0,
    0,  181,    0,   25,    0,
};
final static short yyrindex[] = {                         0,
    0,    0,    0,  223,    0,  107,    0,    0,    0,    0,
  107,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,  176,    0,    0,    0,  203,    0,  203,    0,    0,
    0,  204,    0,    0,    0,    0,    0,    0,    0,    0,
    0,  -58,    0,    0,    0,    0,    0,  -56,    0,    0,
    0,    0,    0,    0,    0,  -28,  -28,  -28,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,  557,   64,    0,    0,  -28,  -58,  -28,  205,
    0,    0,    0,    0,  -28,    0,    0,  -28,   91,  100,
    0,    0,    0,    0,    0,    0,    0,    0,  -28,  -28,
  -28,  -28,  -28,  -28,  -28,  -28,  -28,  -28,  -28,  -28,
  -28,    0,  -28,  -28,   34,    0,    0,    0,    0,  -28,
   59,    0,    0,    0,    0,    0,    0,    0,  113,  -19,
   72,  424,  459,  491,  764,  804,  126,  153,  292,  426,
  751,    0,   -2,  -25,  -58,  -28,  -58,    0,    0,  -28,
    0,    0,  -28,  -28,    0,    0,  231,    0,    0,  -33,
    0,   71,    0,    0,    0,    0,   -3,  -58,    0,    0,
    0,    0,    0,  -58,    0,
};
final static short yygindex[] = {                         0,
    0,  270,    4,   18,    9,  264,  261,    0,  240,    0,
  -27,    0, -134,  -79,    0,    0,    0,    0,    0,    0,
    0,  835,  714,  739,    0,    0,    0,  125,    0,
};
final static int YYTABLESIZE=1088;
static short yytable[];
static { yytable();}
static void yytable(){
yytable = new short[]{                         84,
   41,   33,   86,   33,  119,   33,   84,    1,  127,  117,
  115,   84,  116,  122,  118,   76,  122,   49,   24,   51,
  168,   62,  170,   26,   62,   84,    6,  121,   67,  120,
   30,   24,    2,    7,    9,   68,   26,   41,   39,   62,
   66,   11,  119,  183,   23,   12,   13,  117,  115,  185,
  116,  122,  118,   39,   16,   39,   39,   67,  123,   29,
   31,  123,  119,   50,   68,   46,   36,  117,   94,   66,
   44,  122,  118,   62,   44,   44,   44,   44,   44,   44,
   44,   37,   17,   18,   19,   20,   21,  161,   38,   84,
  160,   84,   44,   44,   44,   44,  123,  182,   45,   78,
   48,   93,   78,   47,   40,   48,   48,   48,   48,   48,
   48,   77,   59,   87,   77,   59,  123,   88,   48,   89,
   69,   95,   40,   48,   44,   48,   44,   64,   96,   97,
   59,   64,   64,   64,   64,   64,   65,   64,   98,  104,
   65,   65,   65,   65,   65,  103,   65,   48,  124,   64,
   64,  105,   64,   61,   48,  106,   61,  125,   65,   65,
  107,   65,   52,  108,   59,  130,   52,   52,   52,   52,
   52,   61,   52,  129,  136,  133,   17,   18,   19,   20,
   21,  134,  138,   64,   52,   52,   32,   52,   35,   53,
   44,  154,   65,   53,   53,   53,   53,   53,   22,   53,
  156,  158,  163,  164,  173,   61,   17,   18,   19,   20,
   21,   53,   53,  176,   53,  160,  178,   43,   52,   43,
  180,  184,    1,   84,   84,   84,   84,   84,   84,   15,
   84,   84,   84,   84,    5,   84,   84,   84,   84,   84,
   84,   84,   84,   20,   19,   53,   84,   43,  111,  112,
   43,   84,   17,   18,   19,   20,   21,   53,   62,   54,
   55,   56,   57,   85,   58,   59,   60,   61,   62,   63,
   64,   75,   43,    8,   14,   65,   28,   43,  167,    0,
   70,   17,   18,   19,   20,   21,   53,    0,   54,   55,
   56,   57,    0,   58,   59,   60,   61,   62,   63,   64,
    0,    0,    0,    0,   65,    0,    0,    0,    0,   70,
   44,   44,    0,    0,   44,   44,   44,   44,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,   54,    0,
    0,    0,   54,   54,   54,   54,   54,    0,   54,    0,
   48,   48,    0,    0,   48,   48,   48,   48,   59,   59,
   54,   54,    0,   54,   59,   59,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,   64,   64,    0,
    0,   64,   64,   64,   64,    0,   65,   65,    0,    0,
   65,   65,   65,   65,   54,    0,    0,  119,    0,   61,
   61,  137,  117,  115,    0,  116,  122,  118,    0,    0,
    0,    0,   52,   52,    0,    0,   52,   52,   52,   52,
  121,    0,  120,    0,  119,    0,    0,    0,  155,  117,
  115,    0,  116,  122,  118,    0,    0,    0,    0,   53,
   53,    0,    0,   53,   53,   53,   53,  121,  119,  120,
    0,  123,  157,  117,  115,    0,  116,  122,  118,  119,
    0,    0,   34,    0,  117,  115,  162,  116,  122,  118,
    0,  121,    0,  120,   60,    0,   58,   60,  123,   58,
  119,    0,  121,    0,  120,  117,  115,    0,  116,  122,
  118,  119,   60,    0,   58,    0,  117,  115,    0,  116,
  122,  118,  123,  121,    0,  120,    0,    0,    0,   55,
    0,    0,   55,  123,  121,  119,  120,    0,    0,    0,
  117,  115,    0,  116,  122,  118,   60,   55,   58,    0,
    0,    0,    0,    0,  123,    0,  165,  177,  121,    0,
  120,   56,    0,   67,   56,  123,    0,  171,  119,    0,
   68,    0,  181,  117,  115,   66,  116,  122,  118,   56,
    0,   55,    0,    0,    0,    0,    0,    0,   67,  123,
    0,  121,    0,  120,    0,   68,    0,    0,   54,   54,
   66,  119,   54,   54,   54,   54,  117,  115,    0,  116,
  122,  118,    0,   56,   17,   18,   19,   20,   21,    0,
    0,    0,  123,   47,  121,    0,  120,    0,   47,   47,
    0,   47,   47,   47,    0,    0,   22,    0,    0,    0,
    0,    0,    0,    0,  119,    0,   47,    0,   47,  117,
  115,    0,  116,  122,  118,  123,    0,  109,  110,    0,
    0,  111,  112,  113,  114,    0,    0,  121,   67,  120,
    0,    0,    0,    0,    0,   68,    0,   47,    0,    0,
   66,    0,    0,    0,  109,  110,    0,    0,  111,  112,
  113,  114,    0,    0,    0,    0,    0,    0,  123,    0,
    0,    0,    0,    0,    0,    0,    0,    0,  109,  110,
    0,    0,  111,  112,  113,  114,    0,    0,    0,  109,
  110,    0,    0,  111,  112,  113,  114,    0,   37,    0,
   60,   60,   58,   58,    0,    0,   60,   60,   58,   58,
  109,  110,    0,    0,  111,  112,  113,  114,    0,    0,
    0,  109,  110,    0,    0,  111,  112,  113,  114,    0,
    0,    0,  119,    0,    0,   55,   55,  117,  115,    0,
  116,  122,  118,    0,    0,  109,  110,    0,    0,  111,
  112,  113,  114,    0,    0,  121,    0,  120,    0,    0,
    0,  101,   53,    0,   54,   83,    0,   56,   56,    0,
    0,   60,    0,   62,   63,   64,    0,    0,  109,  110,
   65,    0,  111,  112,  113,  114,  123,   53,    0,   54,
   84,   57,    0,    0,   57,    0,   60,    0,   62,   63,
   64,   83,    0,    0,   50,   65,   50,   50,   50,   57,
    0,  109,  110,    0,    0,  111,  112,  113,  114,    0,
    0,    0,   50,   50,    0,   50,   84,    0,    0,    0,
    0,    0,    0,   47,   47,    0,    0,   47,   47,   47,
   47,    0,    0,   57,   51,    0,   51,   51,   51,    0,
    0,    0,    0,    0,  109,    0,   50,    0,  111,  112,
  113,  114,   51,   51,    0,   51,    0,   53,   83,   54,
   83,    0,    0,    0,    0,    0,   60,    0,   62,   63,
   64,    0,    0,    0,    0,   65,    0,    0,    0,    0,
   83,   83,   90,   84,    0,   84,   51,   83,    0,    0,
   99,  100,  102,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,   84,   84,    0,    0,    0,
    0,  126,   84,  128,    0,    0,    0,    0,    0,  131,
    0,    0,  135,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,  139,  140,  141,  142,  143,  144,  145,
  146,  147,  148,  149,  150,  151,    0,  152,  153,    0,
    0,    0,    0,    0,  159,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,  111,  112,  113,  114,
    0,    0,    0,    0,    0,    0,    0,    0,  131,    0,
  169,    0,    0,    0,  172,    0,    0,  174,  175,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,   57,   57,    0,
    0,    0,    0,   57,   57,    0,    0,    0,    0,    0,
   50,   50,    0,    0,   50,   50,   50,   50,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
   51,   51,    0,    0,   51,   51,   51,   51,
};
}
static short yycheck[];
static { yycheck(); }
static void yycheck() {
yycheck = new short[] {                         33,
   59,   91,   59,   91,   37,   91,   40,  261,   88,   42,
   43,   45,   45,   46,   47,   41,   46,   45,   15,   47,
  155,   41,  157,   15,   44,   59,  276,   60,   33,   62,
   22,   28,  286,  261,  263,   40,   28,   41,   41,   59,
   45,  276,   37,  178,  125,  276,  123,   42,   43,  184,
   45,   46,   47,   36,  123,   38,   59,   33,   91,  276,
   59,   91,   37,   46,   40,   44,   40,   42,   60,   45,
   37,   46,   47,   93,   41,   42,   43,   44,   45,   46,
   47,   93,  257,  258,  259,  260,  261,   41,   40,  123,
   44,  125,   59,   60,   61,   62,   91,  177,   41,   41,
   37,  276,   44,   41,   41,   42,   43,  123,   45,   46,
   47,   41,   41,   40,   44,   44,   91,   40,  123,   40,
  125,   40,   59,   60,   91,   62,   93,   37,   40,   40,
   59,   41,   42,   43,   44,   45,   37,   47,   40,   59,
   41,   42,   43,   44,   45,   40,   47,  123,   61,   59,
   60,   59,   62,   41,   91,   59,   44,  276,   59,   60,
   59,   62,   37,   59,   93,   91,   41,   42,   43,   44,
   45,   59,   47,   40,  276,   41,  257,  258,  259,  260,
  261,   41,  276,   93,   59,   60,  276,   62,  276,   37,
  276,   40,   93,   41,   42,   43,   44,   45,  279,   47,
   59,   41,   41,   44,  276,   93,  257,  258,  259,  260,
  261,   59,   60,   41,   62,   44,  268,  276,   93,  276,
   41,   41,    0,  257,  258,  259,  260,  261,  262,  123,
  264,  265,  266,  267,   59,  269,  270,  271,  272,  273,
  274,  275,  276,   41,   41,   93,  280,  276,  281,  282,
  276,  285,  257,  258,  259,  260,  261,  262,  278,  264,
  265,  266,  267,   59,  269,  270,  271,  272,  273,  274,
  275,   41,  276,    4,   11,  280,   16,   38,  154,   -1,
  285,  257,  258,  259,  260,  261,  262,   -1,  264,  265,
  266,  267,   -1,  269,  270,  271,  272,  273,  274,  275,
   -1,   -1,   -1,   -1,  280,   -1,   -1,   -1,   -1,  285,
  277,  278,   -1,   -1,  281,  282,  283,  284,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   37,   -1,
   -1,   -1,   41,   42,   43,   44,   45,   -1,   47,   -1,
  277,  278,   -1,   -1,  281,  282,  283,  284,  277,  278,
   59,   60,   -1,   62,  283,  284,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,  277,  278,   -1,
   -1,  281,  282,  283,  284,   -1,  277,  278,   -1,   -1,
  281,  282,  283,  284,   93,   -1,   -1,   37,   -1,  277,
  278,   41,   42,   43,   -1,   45,   46,   47,   -1,   -1,
   -1,   -1,  277,  278,   -1,   -1,  281,  282,  283,  284,
   60,   -1,   62,   -1,   37,   -1,   -1,   -1,   41,   42,
   43,   -1,   45,   46,   47,   -1,   -1,   -1,   -1,  277,
  278,   -1,   -1,  281,  282,  283,  284,   60,   37,   62,
   -1,   91,   41,   42,   43,   -1,   45,   46,   47,   37,
   -1,   -1,  125,   -1,   42,   43,   44,   45,   46,   47,
   -1,   60,   -1,   62,   41,   -1,   41,   44,   91,   44,
   37,   -1,   60,   -1,   62,   42,   43,   -1,   45,   46,
   47,   37,   59,   -1,   59,   -1,   42,   43,   -1,   45,
   46,   47,   91,   60,   -1,   62,   -1,   -1,   -1,   41,
   -1,   -1,   44,   91,   60,   37,   62,   -1,   -1,   -1,
   42,   43,   -1,   45,   46,   47,   93,   59,   93,   -1,
   -1,   -1,   -1,   -1,   91,   -1,   93,   59,   60,   -1,
   62,   41,   -1,   33,   44,   91,   -1,   93,   37,   -1,
   40,   -1,   41,   42,   43,   45,   45,   46,   47,   59,
   -1,   93,   -1,   -1,   -1,   -1,   -1,   -1,   33,   91,
   -1,   60,   -1,   62,   -1,   40,   -1,   -1,  277,  278,
   45,   37,  281,  282,  283,  284,   42,   43,   -1,   45,
   46,   47,   -1,   93,  257,  258,  259,  260,  261,   -1,
   -1,   -1,   91,   37,   60,   -1,   62,   -1,   42,   43,
   -1,   45,   46,   47,   -1,   -1,  279,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   37,   -1,   60,   -1,   62,   42,
   43,   -1,   45,   46,   47,   91,   -1,  277,  278,   -1,
   -1,  281,  282,  283,  284,   -1,   -1,   60,   33,   62,
   -1,   -1,   -1,   -1,   -1,   40,   -1,   91,   -1,   -1,
   45,   -1,   -1,   -1,  277,  278,   -1,   -1,  281,  282,
  283,  284,   -1,   -1,   -1,   -1,   -1,   -1,   91,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,  277,  278,
   -1,   -1,  281,  282,  283,  284,   -1,   -1,   -1,  277,
  278,   -1,   -1,  281,  282,  283,  284,   -1,   93,   -1,
  277,  278,  277,  278,   -1,   -1,  283,  284,  283,  284,
  277,  278,   -1,   -1,  281,  282,  283,  284,   -1,   -1,
   -1,  277,  278,   -1,   -1,  281,  282,  283,  284,   -1,
   -1,   -1,   37,   -1,   -1,  277,  278,   42,   43,   -1,
   45,   46,   47,   -1,   -1,  277,  278,   -1,   -1,  281,
  282,  283,  284,   -1,   -1,   60,   -1,   62,   -1,   -1,
   -1,  261,  262,   -1,  264,   52,   -1,  277,  278,   -1,
   -1,  271,   -1,  273,  274,  275,   -1,   -1,  277,  278,
  280,   -1,  281,  282,  283,  284,   91,  262,   -1,  264,
   52,   41,   -1,   -1,   44,   -1,  271,   -1,  273,  274,
  275,   88,   -1,   -1,   41,  280,   43,   44,   45,   59,
   -1,  277,  278,   -1,   -1,  281,  282,  283,  284,   -1,
   -1,   -1,   59,   60,   -1,   62,   88,   -1,   -1,   -1,
   -1,   -1,   -1,  277,  278,   -1,   -1,  281,  282,  283,
  284,   -1,   -1,   93,   41,   -1,   43,   44,   45,   -1,
   -1,   -1,   -1,   -1,  277,   -1,   93,   -1,  281,  282,
  283,  284,   59,   60,   -1,   62,   -1,  262,  155,  264,
  157,   -1,   -1,   -1,   -1,   -1,  271,   -1,  273,  274,
  275,   -1,   -1,   -1,   -1,  280,   -1,   -1,   -1,   -1,
  177,  178,   58,  155,   -1,  157,   93,  184,   -1,   -1,
   66,   67,   68,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,  177,  178,   -1,   -1,   -1,
   -1,   87,  184,   89,   -1,   -1,   -1,   -1,   -1,   95,
   -1,   -1,   98,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,  109,  110,  111,  112,  113,  114,  115,
  116,  117,  118,  119,  120,  121,   -1,  123,  124,   -1,
   -1,   -1,   -1,   -1,  130,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,  281,  282,  283,  284,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,  154,   -1,
  156,   -1,   -1,   -1,  160,   -1,   -1,  163,  164,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,  277,  278,   -1,
   -1,   -1,   -1,  283,  284,   -1,   -1,   -1,   -1,   -1,
  277,  278,   -1,   -1,  281,  282,  283,  284,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
  277,  278,   -1,   -1,  281,  282,  283,  284,
};
}
final static short YYFINAL=3;
final static short YYMAXTOKEN=288;
final static String yyname[] = {
"end-of-file",null,null,null,null,null,null,null,null,null,null,null,null,null,
null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,
null,null,null,"'!'",null,null,null,"'%'",null,null,"'('","')'","'*'","'+'",
"','","'-'","'.'","'/'",null,null,null,null,null,null,null,null,null,null,null,
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
"SEALED","UMINUS","EMPTY",
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
"Stmt : StmtBlock",
"OCStmt : SCOPY '(' IDENTIFIER ',' Expr ')'",
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

//#line 435 "Parser.y"
    
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
//#line 602 "Parser.java"
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
//#line 53 "Parser.y"
{
						tree = new Tree.TopLevel(val_peek(0).clist, val_peek(0).loc);
					}
break;
case 2:
//#line 59 "Parser.y"
{
						yyval.clist.add(val_peek(0).cdef);
					}
break;
case 3:
//#line 63 "Parser.y"
{
                		yyval.clist = new ArrayList<Tree.ClassDef>();
                		yyval.clist.add(val_peek(0).cdef);
                	}
break;
case 5:
//#line 73 "Parser.y"
{
						yyval.vdef = new Tree.VarDef(val_peek(0).ident, val_peek(1).type, val_peek(0).loc);
					}
break;
case 6:
//#line 79 "Parser.y"
{
						yyval.type = new Tree.TypeIdent(Tree.INT, val_peek(0).loc);
					}
break;
case 7:
//#line 83 "Parser.y"
{
                		yyval.type = new Tree.TypeIdent(Tree.VOID, val_peek(0).loc);
                	}
break;
case 8:
//#line 87 "Parser.y"
{
                		yyval.type = new Tree.TypeIdent(Tree.BOOL, val_peek(0).loc);
                	}
break;
case 9:
//#line 91 "Parser.y"
{
                		yyval.type = new Tree.TypeIdent(Tree.STRING, val_peek(0).loc);
                	}
break;
case 10:
//#line 95 "Parser.y"
{
                		yyval.type = new Tree.TypeClass(val_peek(0).ident, val_peek(1).loc);
                	}
break;
case 11:
//#line 99 "Parser.y"
{
                		yyval.type = new Tree.TypeArray(val_peek(2).type, val_peek(2).loc);
                	}
break;
case 12:
//#line 105 "Parser.y"
{
						yyval.cdef = new Tree.ClassDef(true, val_peek(4).ident, val_peek(3).ident, val_peek(1).flist, val_peek(5).loc);
					}
break;
case 13:
//#line 109 "Parser.y"
{
						yyval.cdef = new Tree.ClassDef(false, val_peek(4).ident, val_peek(3).ident, val_peek(1).flist, val_peek(5).loc);
					}
break;
case 14:
//#line 115 "Parser.y"
{
						yyval.ident = val_peek(0).ident;
					}
break;
case 15:
//#line 119 "Parser.y"
{
                		yyval = new SemValue();
                	}
break;
case 16:
//#line 125 "Parser.y"
{
						yyval.flist.add(val_peek(0).vdef);
					}
break;
case 17:
//#line 129 "Parser.y"
{
						yyval.flist.add(val_peek(0).fdef);
					}
break;
case 18:
//#line 133 "Parser.y"
{
                		yyval = new SemValue();
                		yyval.flist = new ArrayList<Tree>();
                	}
break;
case 20:
//#line 141 "Parser.y"
{
                		yyval = new SemValue();
                		yyval.vlist = new ArrayList<Tree.VarDef>(); 
                	}
break;
case 21:
//#line 148 "Parser.y"
{
						yyval.vlist.add(val_peek(0).vdef);
					}
break;
case 22:
//#line 152 "Parser.y"
{
                		yyval.vlist = new ArrayList<Tree.VarDef>();
						yyval.vlist.add(val_peek(0).vdef);
                	}
break;
case 23:
//#line 159 "Parser.y"
{
						yyval.fdef = new MethodDef(true, val_peek(4).ident, val_peek(5).type, val_peek(2).vlist, (Block) val_peek(0).stmt, val_peek(4).loc);
					}
break;
case 24:
//#line 163 "Parser.y"
{
						yyval.fdef = new MethodDef(false, val_peek(4).ident, val_peek(5).type, val_peek(2).vlist, (Block) val_peek(0).stmt, val_peek(4).loc);
					}
break;
case 25:
//#line 169 "Parser.y"
{
						yyval.stmt = new Block(val_peek(1).slist, val_peek(2).loc);
					}
break;
case 26:
//#line 175 "Parser.y"
{
						yyval.slist.add(val_peek(0).stmt);
					}
break;
case 27:
//#line 179 "Parser.y"
{
                		yyval = new SemValue();
                		yyval.slist = new ArrayList<Tree>();
                	}
break;
case 28:
//#line 186 "Parser.y"
{
						yyval.stmt = val_peek(0).vdef;
					}
break;
case 29:
//#line 191 "Parser.y"
{
                		if (yyval.stmt == null) {
                			yyval.stmt = new Tree.Skip(val_peek(0).loc);
                		}
                	}
break;
case 38:
//#line 207 "Parser.y"
{
						yyval.stmt = new Tree.Scopy(val_peek(1).expr, val_peek(3).ident, val_peek(5).loc);
					}
break;
case 39:
//#line 213 "Parser.y"
{
						yyval.stmt = new Tree.Assign(val_peek(2).lvalue, val_peek(0).expr, val_peek(1).loc);
					}
break;
case 40:
//#line 217 "Parser.y"
{
                		yyval.stmt = new Tree.Exec(val_peek(0).expr, val_peek(0).loc);
                	}
break;
case 41:
//#line 221 "Parser.y"
{
                		yyval = new SemValue();
                	}
break;
case 43:
//#line 228 "Parser.y"
{
                		yyval = new SemValue();
                	}
break;
case 44:
//#line 234 "Parser.y"
{
						yyval.lvalue = new Tree.Ident(val_peek(1).expr, val_peek(0).ident, val_peek(0).loc);
						if (val_peek(1).loc == null) {
							yyval.loc = val_peek(0).loc;
						}
					}
break;
case 45:
//#line 241 "Parser.y"
{
                		yyval.lvalue = new Tree.Indexed(val_peek(3).expr, val_peek(1).expr, val_peek(3).loc);
                	}
break;
case 46:
//#line 247 "Parser.y"
{
						yyval.expr = new Tree.CallExpr(val_peek(4).expr, val_peek(3).ident, val_peek(1).elist, val_peek(3).loc);
						if (val_peek(4).loc == null) {
							yyval.loc = val_peek(3).loc;
						}
					}
break;
case 47:
//#line 256 "Parser.y"
{
						yyval.expr = val_peek(0).lvalue;
					}
break;
case 50:
//#line 262 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.PLUS, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 51:
//#line 266 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.MINUS, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 52:
//#line 270 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.MUL, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 53:
//#line 274 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.DIV, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 54:
//#line 278 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.MOD, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 55:
//#line 282 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.EQ, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 56:
//#line 286 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.NE, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 57:
//#line 290 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.LT, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 58:
//#line 294 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.GT, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 59:
//#line 298 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.LE, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 60:
//#line 302 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.GE, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 61:
//#line 306 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.AND, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 62:
//#line 310 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.OR, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 63:
//#line 314 "Parser.y"
{
                		yyval = val_peek(1);
                	}
break;
case 64:
//#line 318 "Parser.y"
{
                		yyval.expr = new Tree.Unary(Tree.NEG, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 65:
//#line 322 "Parser.y"
{
                		yyval.expr = new Tree.Unary(Tree.NOT, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 66:
//#line 326 "Parser.y"
{
                		yyval.expr = new Tree.ReadIntExpr(val_peek(2).loc);
                	}
break;
case 67:
//#line 330 "Parser.y"
{
                		yyval.expr = new Tree.ReadLineExpr(val_peek(2).loc);
                	}
break;
case 68:
//#line 334 "Parser.y"
{
                		yyval.expr = new Tree.ThisExpr(val_peek(0).loc);
                	}
break;
case 69:
//#line 338 "Parser.y"
{
                		yyval.expr = new Tree.NewClass(val_peek(2).ident, val_peek(3).loc);
                	}
break;
case 70:
//#line 342 "Parser.y"
{
                		yyval.expr = new Tree.NewArray(val_peek(3).type, val_peek(1).expr, val_peek(4).loc);
                	}
break;
case 71:
//#line 346 "Parser.y"
{
                		yyval.expr = new Tree.TypeTest(val_peek(3).expr, val_peek(1).ident, val_peek(5).loc);
                	}
break;
case 72:
//#line 350 "Parser.y"
{
                		yyval.expr = new Tree.TypeCast(val_peek(2).ident, val_peek(0).expr, val_peek(0).loc);
                	}
break;
case 73:
//#line 356 "Parser.y"
{
						yyval.expr = new Tree.Literal(val_peek(0).typeTag, val_peek(0).literal, val_peek(0).loc);
					}
break;
case 74:
//#line 360 "Parser.y"
{
						yyval.expr = new Null(val_peek(0).loc);
					}
break;
case 76:
//#line 367 "Parser.y"
{
                		yyval = new SemValue();
                		yyval.elist = new ArrayList<Tree.Expr>();
                	}
break;
case 77:
//#line 374 "Parser.y"
{
						yyval.elist.add(val_peek(0).expr);
					}
break;
case 78:
//#line 378 "Parser.y"
{
                		yyval.elist = new ArrayList<Tree.Expr>();
						yyval.elist.add(val_peek(0).expr);
                	}
break;
case 79:
//#line 385 "Parser.y"
{
						yyval.stmt = new Tree.WhileLoop(val_peek(2).expr, val_peek(0).stmt, val_peek(4).loc);
					}
break;
case 80:
//#line 391 "Parser.y"
{
						yyval.stmt = new Tree.ForLoop(val_peek(6).stmt, val_peek(4).expr, val_peek(2).stmt, val_peek(0).stmt, val_peek(8).loc);
					}
break;
case 81:
//#line 397 "Parser.y"
{
						yyval.stmt = new Tree.Break(val_peek(0).loc);
					}
break;
case 82:
//#line 403 "Parser.y"
{
						yyval.stmt = new Tree.If(val_peek(3).expr, val_peek(1).stmt, val_peek(0).stmt, val_peek(5).loc);
					}
break;
case 83:
//#line 409 "Parser.y"
{
						yyval.stmt = val_peek(0).stmt;
					}
break;
case 84:
//#line 413 "Parser.y"
{
						yyval = new SemValue();
					}
break;
case 85:
//#line 419 "Parser.y"
{
						yyval.stmt = new Tree.Return(val_peek(0).expr, val_peek(1).loc);
					}
break;
case 86:
//#line 423 "Parser.y"
{
                		yyval.stmt = new Tree.Return(null, val_peek(0).loc);
                	}
break;
case 87:
//#line 429 "Parser.y"
{
						yyval.stmt = new Print(val_peek(1).elist, val_peek(3).loc);
					}
break;
//#line 1201 "Parser.java"
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
