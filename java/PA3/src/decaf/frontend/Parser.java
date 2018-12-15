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
public final static short UMINUS=290;
public final static short EMPTY=291;
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
   23,   31,   31,   30,   30,   32,   32,   16,   17,   20,
   15,   33,   33,   18,   18,   19,
};
final static short yylen[] = {                            2,
    1,    2,    1,    2,    2,    1,    1,    1,    1,    2,
    3,    7,    6,    2,    0,    2,    2,    0,    1,    0,
    3,    1,    7,    6,    3,    2,    0,    1,    2,    1,
    1,    1,    2,    2,    2,    2,    1,    1,    6,    5,
    3,    3,    0,    3,    3,    1,    0,    2,    0,    2,
    4,    1,    2,    5,    1,    1,    1,    3,    3,    3,
    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,
    3,    3,    2,    2,    3,    3,    1,    4,    5,    6,
    5,    1,    1,    1,    0,    3,    1,    5,    9,    1,
    6,    2,    0,    2,    1,    4,
};
final static short yydefred[] = {                         0,
    0,    0,    0,    0,    3,    0,    0,    2,    0,    0,
    0,   14,   18,    0,    0,   18,    7,    8,    6,    9,
    0,    0,   13,   16,    0,    0,   17,    0,   10,    0,
    4,    0,    0,   12,    0,    0,   11,    0,   22,    0,
    0,    0,    0,    5,    0,    0,    0,   27,   24,   21,
   23,    0,   83,   77,    0,    0,    0,    0,   90,    0,
    0,    0,    0,   82,    0,    0,    0,    0,   25,    0,
    0,   28,   38,   26,    0,   30,   31,   32,    0,    0,
    0,    0,   37,    0,    0,    0,    0,   52,   57,    0,
    0,    0,    0,    0,   55,   56,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,   53,   29,   33,
   34,   35,   36,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,   48,    0,    0,    0,
    0,    0,    0,    0,   41,    0,    0,    0,    0,    0,
   75,   76,    0,    0,   72,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,   78,    0,
    0,   96,    0,    0,    0,   51,    0,    0,   88,    0,
    0,    0,   40,   42,   79,    0,    0,   81,    0,   54,
    0,    0,   91,   44,   80,   39,    0,   92,    0,   89,
};
final static short yydgoto[] = {                          3,
    4,    5,   72,   25,   40,   10,   15,   27,   41,   42,
   73,   52,   74,   75,   76,   77,   78,   79,   80,   81,
   82,   83,   84,  136,  168,   95,   96,   87,   88,  177,
   89,  140,  193,
};
final static short yysindex[] = {                      -244,
 -258, -237,    0, -244,    0, -234, -242,    0, -238,  -72,
 -234,    0,    0,  -54,  271,    0,    0,    0,    0,    0,
 -223, -120,    0,    0,   15,  -90,    0,  297,    0,  -89,
    0,   39,  -18,    0,   43, -120,    0, -120,    0,  -70,
   47,   45,   52,    0,  -12, -120,  -12,    0,    0,    0,
    0,   -1,    0,    0,   73,   74,  -37,  110,    0, -203,
   79,   81,   85,    0,   87,  110,  110,   51,    0,   88,
 -178,    0,    0,    0,   46,    0,    0,    0,   53,   75,
   83,   86,    0,  781,   68,    0, -145,    0,    0,  110,
  110,  110,   19,  781,    0,    0,   93,   56,  110,  107,
  108,  110,  -26,  -26, -124,  480, -119,    0,    0,    0,
    0,    0,    0,  110,  110,  110,  110,  110,  110,  110,
  110,  110,  110,  110,  110,  110,    0,  110,  110,  110,
  116,  506,   99,  532,    0,  110,  118,   70,  781,    6,
    0,    0,  556,  120,    0,  128,  868,  857,   35,   35,
  889,  889,   -6,   -6,  -26,  -26,  -26,   35,   35,  582,
  -32,  781,  110,   31,  110,   31,  603, -116,    0,  705,
  110,    0, -114,  110,  110,    0,  132,  130,    0,  760,
  -93,   31,    0,    0,    0,  781,  135,    0,  807,    0,
  110,   31,    0,    0,    0,    0,  136,    0,   31,    0,
};
final static short yyrindex[] = {                         0,
    0,    0,    0,  178,    0,   57,    0,    0,    0,    0,
   57,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,  131,    0,    0,    0,  138,    0,  138,    0,    0,
    0,  148,    0,    0,    0,    0,    0,    0,    0,    0,
    0,  -55,    0,    0,    0,    0,    0,  -53,    0,    0,
    0,    0,    0,    0,    0,  -85,  -85,  -85,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,  831,  454,    0,    0,    0,  -85,
  -55,  -85,   90,  140,    0,    0,    0,    0,  -85,    0,
    0,  -85,  151,  160,    0,    0,    0,    0,    0,    0,
    0,    0,    0,  -85,  -85,  -85,  -85,  -85,  -85,  -85,
  -85,  -85,  -85,  -85,  -85,  -85,    0,  -85,  -85,  -85,
  123,    0,    0,    0,    0,  -85,    0,  -85,   63,    0,
    0,    0,    0,    0,    0,    0,    4,    2,  612,  906,
    8,   58,  898,  952,  409,  418,  445,  926,  962,    0,
  981,  -22,  -25,  -55,  -85,  -55,    0,    0,    0,    0,
  -85,    0,    0,  -85,  -85,    0,    0,  159,    0,    0,
  -33,  -55,    0,    0,    0,   65,    0,    0,    0,    0,
  -14,  -55,    0,    0,    0,    0,    0,    0,  -55,    0,
};
final static short yygindex[] = {                         0,
    0,  204,    7,  -13,   72,  201,  199,    0,  179,    0,
   23,    0,  -46,  -83,    0,    0,    0,    0,    0,    0,
    0,    0, 1158,    0,    0,  836,  963,    0,    0,    0,
    0,   67,    0,
};
final static int YYTABLESIZE=1333;
static short yytable[];
static { yytable();}
static void yytable(){
yytable = new short[]{                         93,
   33,   33,   92,   47,  124,   95,   93,  133,  183,  122,
  120,   93,  121,  127,  123,   85,    1,    6,   45,  127,
   33,   24,   39,    7,   39,   93,   47,  126,    9,  125,
  124,   67,   50,   11,   24,  122,   45,   12,   68,  127,
  123,    2,   71,   66,   70,   71,  172,   70,   63,  171,
   13,   63,   29,   17,   18,   19,   20,   21,  128,   71,
   71,   70,   70,   67,  128,   63,   63,   49,   16,   51,
   68,  124,   97,   31,   37,   66,  122,  120,   36,  121,
  127,  123,   38,   67,  128,   93,   26,   45,   46,   93,
   68,   93,   47,   30,   71,   66,   70,  108,   64,   26,
   63,   64,   67,   87,  109,   86,   87,  197,   86,   68,
   48,  110,   90,   91,   66,   64,   64,  179,   99,  181,
  100,   48,   43,   69,  101,  128,  102,  107,  130,   43,
  131,   98,  137,  111,   43,  194,   17,   18,   19,   20,
   21,  112,   67,  135,  113,  198,  138,  141,  142,   68,
   64,  144,  200,   48,   66,  163,  146,  165,  169,   50,
  174,  187,   37,   50,   50,   50,   50,   50,   50,   50,
  184,  175,  190,  171,  192,  195,  199,    1,   20,   15,
   50,   50,   50,   50,   50,   32,   35,   73,   19,    5,
   49,   73,   73,   73,   73,   73,   74,   73,   94,   84,
   74,   74,   74,   74,   74,   44,   74,    8,   73,   73,
   73,   14,   73,   50,   28,   50,   43,   74,   74,   74,
   49,   74,   49,   93,   93,   93,   93,   93,   93,  178,
   93,   93,   93,   93,    0,   93,   93,   93,   93,   93,
   93,   93,   93,   73,    0,    0,   93,    0,  116,  117,
   49,   93,   74,   93,   93,   17,   18,   19,   20,   21,
   53,   49,   54,   55,   56,   57,    0,   58,   59,   60,
   61,   62,   63,   64,    0,    0,    0,    0,   65,   71,
   70,   70,    0,   70,   63,   63,   71,   17,   18,   19,
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
    0,    0,    0,    0,    0,   23,    0,   71,    0,   50,
   50,    0,    0,   50,   50,   50,   50,    0,    0,    0,
    0,   50,    0,    0,    0,    0,    0,    0,    0,    0,
    0,   34,    0,    0,    0,    0,    0,   73,   73,    0,
    0,   73,   73,   73,   73,    0,   74,   74,    0,   73,
   74,   74,   74,   74,    0,   60,    0,    0,   74,   60,
   60,   60,   60,   60,   61,   60,    0,    0,   61,   61,
   61,   61,   61,    0,   61,    0,   60,   60,   60,    0,
   60,    0,    0,    0,    0,   61,   61,   61,    0,   61,
    0,   62,    0,    0,    0,   62,   62,   62,   62,   62,
   56,   62,    0,    0,   46,   56,   56,    0,   56,   56,
   56,   60,   62,   62,   62,    0,   62,    0,    0,    0,
   61,    0,   46,   56,    0,   56,  124,    0,    0,    0,
  145,  122,  120,    0,  121,  127,  123,   17,   18,   19,
   20,   21,    0,    0,    0,    0,    0,   62,    0,  126,
    0,  125,  124,    0,   56,    0,  164,  122,  120,   22,
  121,  127,  123,   17,   18,   19,   20,   21,    0,    0,
    0,    0,    0,    0,    0,  126,    0,  125,  124,    0,
  128,    0,  166,  122,  120,   22,  121,  127,  123,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,  126,  124,  125,    0,    0,  128,  122,  120,  173,
  121,  127,  123,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,  126,    0,  125,  124,    0,
    0,    0,  128,  122,  120,    0,  121,  127,  123,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,  124,
    0,  126,    0,  125,  122,  120,  128,  121,  127,  123,
    0,    0,   68,    0,    0,   68,    0,    0,    0,    0,
  182,    0,  126,    0,  125,    0,    0,    0,    0,   68,
   68,    0,  128,    0,  176,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,   60,   60,    0,    0,   60,
   60,   60,   60,  128,   61,   61,    0,   60,   61,   61,
   61,   61,    0,    0,   68,    0,   61,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,   62,   62,    0,    0,   62,   62,   62,   62,    0,
   56,   56,    0,   62,   56,   56,   56,   56,    0,    0,
    0,  124,   56,    0,    0,    0,  122,  120,    0,  121,
  127,  123,    0,    0,    0,    0,  114,  115,    0,    0,
  116,  117,  118,  119,  126,    0,  125,    0,  129,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,  114,  115,    0,    0,  116,  117,  118,  119,
    0,    0,    0,    0,  129,  128,  124,  185,    0,    0,
    0,  122,  120,    0,  121,  127,  123,    0,  114,  115,
    0,    0,  116,  117,  118,  119,    0,  124,  191,  126,
  129,  125,  122,  120,    0,  121,  127,  123,    0,    0,
    0,    0,  114,  115,    0,    0,  116,  117,  118,  119,
  126,    0,  125,  124,  129,    0,    0,  196,  122,  120,
  128,  121,  127,  123,    0,    0,    0,    0,  114,  115,
    0,    0,  116,  117,  118,  119,  126,   55,  125,    0,
  129,  128,   55,   55,    0,   55,   55,   55,    0,  114,
  115,    0,    0,  116,  117,  118,  119,   85,   68,   68,
   55,  129,   55,  124,   68,   68,    0,  128,  122,  120,
   68,  121,  127,  123,  124,    0,    0,    0,    0,  122,
  120,    0,  121,  127,  123,    0,  126,    0,  125,    0,
    0,   55,    0,    0,    0,  124,   85,  126,    0,  125,
  122,  120,    0,  121,  127,  123,    0,    0,   58,    0,
   58,   58,   58,    0,    0,    0,   69,  128,  126,   69,
  125,    0,    0,    0,    0,   58,   58,   58,  128,   58,
    0,    0,    0,   69,   69,    0,   66,    0,    0,   66,
    0,    0,    0,    0,    0,    0,    0,    0,    0,  128,
    0,  114,  115,   66,   66,  116,  117,  118,  119,    0,
   58,    0,   59,  129,   59,   59,   59,    0,   69,   85,
    0,   85,   65,    0,    0,   65,    0,    0,    0,   59,
   59,   59,    0,   59,   86,    0,    0,   85,   66,   65,
   65,   67,    0,    0,   67,    0,   85,   85,    0,    0,
    0,    0,    0,    0,   85,    0,  114,  115,   67,   67,
  116,  117,  118,  119,   59,    0,    0,    0,  129,    0,
    0,    0,    0,   86,   65,    0,    0,  114,  115,    0,
    0,  116,  117,  118,  119,    0,    0,    0,    0,  129,
    0,    0,    0,   67,    0,    0,    0,    0,    0,    0,
    0,    0,    0,  114,  115,    0,    0,  116,  117,  118,
  119,    0,    0,    0,    0,  129,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,   55,   55,    0,
    0,   55,   55,   55,   55,    0,    0,    0,    0,   55,
    0,    0,    0,    0,    0,    0,   86,    0,   86,    0,
    0,    0,    0,  114,    0,    0,    0,  116,  117,  118,
  119,    0,    0,    0,   86,  129,    0,    0,  116,  117,
  118,  119,    0,   86,   86,    0,  129,    0,    0,    0,
    0,   86,    0,    0,    0,    0,    0,    0,    0,  116,
  117,    0,    0,    0,   58,   58,    0,  129,   58,   58,
   58,   58,   69,   69,    0,    0,   58,    0,   69,   69,
    0,    0,    0,    0,   69,    0,    0,    0,    0,    0,
    0,    0,   66,   66,    0,    0,    0,    0,   66,   66,
    0,    0,    0,    0,   66,   94,    0,    0,    0,    0,
    0,    0,    0,  103,  104,  106,    0,    0,   59,   59,
    0,    0,   59,   59,   59,   59,    0,    0,   65,   65,
   59,    0,    0,    0,   65,   65,    0,  132,    0,  134,
   65,    0,    0,    0,    0,    0,  139,   67,   67,  143,
    0,    0,    0,   67,   67,    0,    0,    0,    0,   67,
    0,  147,  148,  149,  150,  151,  152,  153,  154,  155,
  156,  157,  158,  159,    0,  160,  161,  162,    0,    0,
    0,    0,    0,  167,    0,  170,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
  139,    0,  180,    0,    0,    0,    0,    0,  186,    0,
    0,  188,  189,
};
}
static short yycheck[];
static { yycheck(); }
static void yycheck() {
yycheck = new short[] {                         33,
   91,   91,   40,   59,   37,   59,   40,   91,  125,   42,
   43,   45,   45,   46,   47,   41,  261,  276,   41,   46,
   91,   15,   36,  261,   38,   59,   41,   60,  263,   62,
   37,   33,   46,  276,   28,   42,   59,  276,   40,   46,
   47,  286,   41,   45,   41,   44,   41,   44,   41,   44,
  123,   44,  276,  257,  258,  259,  260,  261,   91,   58,
   59,   58,   59,   33,   91,   58,   59,   45,  123,   47,
   40,   37,  276,   59,   93,   45,   42,   43,   40,   45,
   46,   47,   40,   33,   91,  123,   15,   41,   44,  123,
   40,  125,   41,   22,   93,   45,   93,  276,   41,   28,
   93,   44,   33,   41,   59,   41,   44,  191,   44,   40,
  123,   59,   40,   40,   45,   58,   59,  164,   40,  166,
   40,  123,   33,  125,   40,   91,   40,   40,   61,   40,
  276,   60,   40,   59,   45,  182,  257,  258,  259,  260,
  261,   59,   33,  125,   59,  192,   91,   41,   41,   40,
   93,  276,  199,  123,   45,   40,  276,   59,   41,   37,
   41,  276,   93,   41,   42,   43,   44,   45,   46,   47,
  287,   44,   41,   44,  268,   41,   41,    0,   41,  123,
   58,   59,   60,   61,   62,  276,  276,   37,   41,   59,
  276,   41,   42,   43,   44,   45,   37,   47,   59,   41,
   41,   42,   43,   44,   45,  276,   47,    4,   58,   59,
   60,   11,   62,   91,   16,   93,   38,   58,   59,   60,
  276,   62,  276,  257,  258,  259,  260,  261,  262,  163,
  264,  265,  266,  267,   -1,  269,  270,  271,  272,  273,
  274,  275,  276,   93,   -1,   -1,  280,   -1,  281,  282,
  276,  285,   93,  287,  288,  257,  258,  259,  260,  261,
  262,  276,  264,  265,  266,  267,   -1,  269,  270,  271,
  272,  273,  274,  275,   -1,   -1,   -1,   -1,  280,  278,
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
   -1,   -1,   -1,   -1,   -1,  125,   -1,  288,   -1,  277,
  278,   -1,   -1,  281,  282,  283,  284,   -1,   -1,   -1,
   -1,  289,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,  125,   -1,   -1,   -1,   -1,   -1,  277,  278,   -1,
   -1,  281,  282,  283,  284,   -1,  277,  278,   -1,  289,
  281,  282,  283,  284,   -1,   37,   -1,   -1,  289,   41,
   42,   43,   44,   45,   37,   47,   -1,   -1,   41,   42,
   43,   44,   45,   -1,   47,   -1,   58,   59,   60,   -1,
   62,   -1,   -1,   -1,   -1,   58,   59,   60,   -1,   62,
   -1,   37,   -1,   -1,   -1,   41,   42,   43,   44,   45,
   37,   47,   -1,   -1,   41,   42,   43,   -1,   45,   46,
   47,   93,   58,   59,   60,   -1,   62,   -1,   -1,   -1,
   93,   -1,   59,   60,   -1,   62,   37,   -1,   -1,   -1,
   41,   42,   43,   -1,   45,   46,   47,  257,  258,  259,
  260,  261,   -1,   -1,   -1,   -1,   -1,   93,   -1,   60,
   -1,   62,   37,   -1,   91,   -1,   41,   42,   43,  279,
   45,   46,   47,  257,  258,  259,  260,  261,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   60,   -1,   62,   37,   -1,
   91,   -1,   41,   42,   43,  279,   45,   46,   47,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   60,   37,   62,   -1,   -1,   91,   42,   43,   44,
   45,   46,   47,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   60,   -1,   62,   37,   -1,
   -1,   -1,   91,   42,   43,   -1,   45,   46,   47,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   37,
   -1,   60,   -1,   62,   42,   43,   91,   45,   46,   47,
   -1,   -1,   41,   -1,   -1,   44,   -1,   -1,   -1,   -1,
   58,   -1,   60,   -1,   62,   -1,   -1,   -1,   -1,   58,
   59,   -1,   91,   -1,   93,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,  277,  278,   -1,   -1,  281,
  282,  283,  284,   91,  277,  278,   -1,  289,  281,  282,
  283,  284,   -1,   -1,   93,   -1,  289,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,  277,  278,   -1,   -1,  281,  282,  283,  284,   -1,
  277,  278,   -1,  289,  281,  282,  283,  284,   -1,   -1,
   -1,   37,  289,   -1,   -1,   -1,   42,   43,   -1,   45,
   46,   47,   -1,   -1,   -1,   -1,  277,  278,   -1,   -1,
  281,  282,  283,  284,   60,   -1,   62,   -1,  289,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,  277,  278,   -1,   -1,  281,  282,  283,  284,
   -1,   -1,   -1,   -1,  289,   91,   37,   93,   -1,   -1,
   -1,   42,   43,   -1,   45,   46,   47,   -1,  277,  278,
   -1,   -1,  281,  282,  283,  284,   -1,   37,   59,   60,
  289,   62,   42,   43,   -1,   45,   46,   47,   -1,   -1,
   -1,   -1,  277,  278,   -1,   -1,  281,  282,  283,  284,
   60,   -1,   62,   37,  289,   -1,   -1,   41,   42,   43,
   91,   45,   46,   47,   -1,   -1,   -1,   -1,  277,  278,
   -1,   -1,  281,  282,  283,  284,   60,   37,   62,   -1,
  289,   91,   42,   43,   -1,   45,   46,   47,   -1,  277,
  278,   -1,   -1,  281,  282,  283,  284,   52,  277,  278,
   60,  289,   62,   37,  283,  284,   -1,   91,   42,   43,
  289,   45,   46,   47,   37,   -1,   -1,   -1,   -1,   42,
   43,   -1,   45,   46,   47,   -1,   60,   -1,   62,   -1,
   -1,   91,   -1,   -1,   -1,   37,   91,   60,   -1,   62,
   42,   43,   -1,   45,   46,   47,   -1,   -1,   41,   -1,
   43,   44,   45,   -1,   -1,   -1,   41,   91,   60,   44,
   62,   -1,   -1,   -1,   -1,   58,   59,   60,   91,   62,
   -1,   -1,   -1,   58,   59,   -1,   41,   -1,   -1,   44,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   91,
   -1,  277,  278,   58,   59,  281,  282,  283,  284,   -1,
   93,   -1,   41,  289,   43,   44,   45,   -1,   93,  164,
   -1,  166,   41,   -1,   -1,   44,   -1,   -1,   -1,   58,
   59,   60,   -1,   62,   52,   -1,   -1,  182,   93,   58,
   59,   41,   -1,   -1,   44,   -1,  191,  192,   -1,   -1,
   -1,   -1,   -1,   -1,  199,   -1,  277,  278,   58,   59,
  281,  282,  283,  284,   93,   -1,   -1,   -1,  289,   -1,
   -1,   -1,   -1,   91,   93,   -1,   -1,  277,  278,   -1,
   -1,  281,  282,  283,  284,   -1,   -1,   -1,   -1,  289,
   -1,   -1,   -1,   93,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,  277,  278,   -1,   -1,  281,  282,  283,
  284,   -1,   -1,   -1,   -1,  289,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,  277,  278,   -1,
   -1,  281,  282,  283,  284,   -1,   -1,   -1,   -1,  289,
   -1,   -1,   -1,   -1,   -1,   -1,  164,   -1,  166,   -1,
   -1,   -1,   -1,  277,   -1,   -1,   -1,  281,  282,  283,
  284,   -1,   -1,   -1,  182,  289,   -1,   -1,  281,  282,
  283,  284,   -1,  191,  192,   -1,  289,   -1,   -1,   -1,
   -1,  199,   -1,   -1,   -1,   -1,   -1,   -1,   -1,  281,
  282,   -1,   -1,   -1,  277,  278,   -1,  289,  281,  282,
  283,  284,  277,  278,   -1,   -1,  289,   -1,  283,  284,
   -1,   -1,   -1,   -1,  289,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,  277,  278,   -1,   -1,   -1,   -1,  283,  284,
   -1,   -1,   -1,   -1,  289,   58,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   66,   67,   68,   -1,   -1,  277,  278,
   -1,   -1,  281,  282,  283,  284,   -1,   -1,  277,  278,
  289,   -1,   -1,   -1,  283,  284,   -1,   90,   -1,   92,
  289,   -1,   -1,   -1,   -1,   -1,   99,  277,  278,  102,
   -1,   -1,   -1,  283,  284,   -1,   -1,   -1,   -1,  289,
   -1,  114,  115,  116,  117,  118,  119,  120,  121,  122,
  123,  124,  125,  126,   -1,  128,  129,  130,   -1,   -1,
   -1,   -1,   -1,  136,   -1,  138,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
  163,   -1,  165,   -1,   -1,   -1,   -1,   -1,  171,   -1,
   -1,  174,  175,
};
}
final static short YYFINAL=3;
final static short YYMAXTOKEN=291;
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
"SEALED","SEP","VAR","INIT","UMINUS","EMPTY",
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

//#line 476 "Parser.y"
    
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
//#line 671 "Parser.java"
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
                		yyval.expr = new Tree.Binary(Tree.LE, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 69:
//#line 343 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.GE, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 70:
//#line 347 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.AND, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 71:
//#line 351 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.OR, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 72:
//#line 355 "Parser.y"
{
                		yyval = val_peek(1);
                	}
break;
case 73:
//#line 359 "Parser.y"
{
                		yyval.expr = new Tree.Unary(Tree.NEG, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 74:
//#line 363 "Parser.y"
{
                		yyval.expr = new Tree.Unary(Tree.NOT, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 75:
//#line 367 "Parser.y"
{
                		yyval.expr = new Tree.ReadIntExpr(val_peek(2).loc);
                	}
break;
case 76:
//#line 371 "Parser.y"
{
                		yyval.expr = new Tree.ReadLineExpr(val_peek(2).loc);
                	}
break;
case 77:
//#line 375 "Parser.y"
{
                		yyval.expr = new Tree.ThisExpr(val_peek(0).loc);
                	}
break;
case 78:
//#line 379 "Parser.y"
{
                		yyval.expr = new Tree.NewClass(val_peek(2).ident, val_peek(3).loc);
                	}
break;
case 79:
//#line 383 "Parser.y"
{
                		yyval.expr = new Tree.NewArray(val_peek(3).type, val_peek(1).expr, val_peek(4).loc);
                	}
break;
case 80:
//#line 387 "Parser.y"
{
                		yyval.expr = new Tree.TypeTest(val_peek(3).expr, val_peek(1).ident, val_peek(5).loc);
                	}
break;
case 81:
//#line 391 "Parser.y"
{
                		yyval.expr = new Tree.TypeCast(val_peek(2).ident, val_peek(0).expr, val_peek(0).loc);
                	}
break;
case 82:
//#line 397 "Parser.y"
{
						yyval.expr = new Tree.Literal(val_peek(0).typeTag, val_peek(0).literal, val_peek(0).loc);
					}
break;
case 83:
//#line 401 "Parser.y"
{
						yyval.expr = new Null(val_peek(0).loc);
					}
break;
case 85:
//#line 408 "Parser.y"
{
                		yyval = new SemValue();
                		yyval.elist = new ArrayList<Tree.Expr>();
                	}
break;
case 86:
//#line 415 "Parser.y"
{
						yyval.elist.add(val_peek(0).expr);
					}
break;
case 87:
//#line 419 "Parser.y"
{
                		yyval.elist = new ArrayList<Tree.Expr>();
						yyval.elist.add(val_peek(0).expr);
                	}
break;
case 88:
//#line 426 "Parser.y"
{
						yyval.stmt = new Tree.WhileLoop(val_peek(2).expr, val_peek(0).stmt, val_peek(4).loc);
					}
break;
case 89:
//#line 432 "Parser.y"
{
						yyval.stmt = new Tree.ForLoop(val_peek(6).stmt, val_peek(4).expr, val_peek(2).stmt, val_peek(0).stmt, val_peek(8).loc);
					}
break;
case 90:
//#line 438 "Parser.y"
{
						yyval.stmt = new Tree.Break(val_peek(0).loc);
					}
break;
case 91:
//#line 444 "Parser.y"
{
						yyval.stmt = new Tree.If(val_peek(3).expr, val_peek(1).stmt, val_peek(0).stmt, val_peek(5).loc);
					}
break;
case 92:
//#line 450 "Parser.y"
{
						yyval.stmt = val_peek(0).stmt;
					}
break;
case 93:
//#line 454 "Parser.y"
{
						yyval = new SemValue();
					}
break;
case 94:
//#line 460 "Parser.y"
{
						yyval.stmt = new Tree.Return(val_peek(0).expr, val_peek(1).loc);
					}
break;
case 95:
//#line 464 "Parser.y"
{
                		yyval.stmt = new Tree.Return(null, val_peek(0).loc);
                	}
break;
case 96:
//#line 470 "Parser.y"
{
						yyval.stmt = new Print(val_peek(1).elist, val_peek(3).loc);
					}
break;
//#line 1314 "Parser.java"
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
