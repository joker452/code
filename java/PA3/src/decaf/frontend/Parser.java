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
public final static short UMINUS=289;
public final static short EMPTY=290;
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
   31,   31,   30,   30,   32,   32,   16,   17,   20,   15,
   33,   33,   18,   18,   19,
};
final static short yylen[] = {                            2,
    1,    2,    1,    2,    2,    1,    1,    1,    1,    2,
    3,    7,    6,    2,    0,    2,    2,    0,    1,    0,
    3,    1,    7,    6,    3,    2,    0,    1,    2,    1,
    1,    1,    2,    2,    2,    2,    1,    1,    6,    5,
    3,    3,    0,    3,    3,    1,    0,    2,    0,    2,
    4,    1,    2,    5,    1,    1,    1,    3,    3,    3,
    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,
    3,    2,    2,    3,    3,    1,    4,    5,    6,    5,
    1,    1,    1,    0,    3,    1,    5,    9,    1,    6,
    2,    0,    2,    1,    4,
};
final static short yydefred[] = {                         0,
    0,    0,    0,    0,    3,    0,    0,    2,    0,    0,
    0,   14,   18,    0,    0,   18,    7,    8,    6,    9,
    0,    0,   13,   16,    0,    0,   17,    0,   10,    0,
    4,    0,    0,   12,    0,    0,   11,    0,   22,    0,
    0,    0,    0,    5,    0,    0,    0,   27,   24,   21,
   23,    0,   82,   76,    0,    0,    0,    0,   89,    0,
    0,    0,    0,   81,    0,    0,    0,    0,   25,    0,
    0,   28,   38,   26,    0,   30,   31,   32,    0,    0,
    0,    0,   37,    0,    0,    0,    0,   52,   57,    0,
    0,    0,    0,    0,   55,   56,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,   53,   29,   33,
   34,   35,   36,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,   48,    0,    0,    0,
    0,    0,    0,   41,    0,    0,    0,    0,    0,   74,
   75,    0,    0,   71,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,   77,    0,    0,   95,
    0,    0,    0,   51,    0,    0,   87,    0,    0,    0,
   40,   42,   78,    0,    0,   80,    0,   54,    0,    0,
   90,   44,   79,   39,    0,   91,    0,   88,
};
final static short yydgoto[] = {                          3,
    4,    5,   72,   25,   40,   10,   15,   27,   41,   42,
   73,   52,   74,   75,   76,   77,   78,   79,   80,   81,
   82,   83,   84,  135,  166,   95,   96,   87,   88,  175,
   89,  139,  191,
};
final static short yysindex[] = {                      -244,
 -258, -237,    0, -244,    0, -238, -247,    0, -243,  -85,
 -238,    0,    0,  -72,  323,    0,    0,    0,    0,    0,
 -242, -120,    0,    0,   10,  -90,    0,  622,    0,  -89,
    0,   34,  -18,    0,   39, -120,    0, -120,    0,  -70,
   42,   44,   48,    0,  -30, -120,  -30,    0,    0,    0,
    0,   -1,    0,    0,   67,   69,  -37,  110,    0, -203,
   71,   73,   79,    0,   87,  110,  110,   51,    0,   89,
 -178,    0,    0,    0,   53,    0,    0,    0,   74,   75,
   83,   85,    0,  760,   86,    0, -145,    0,    0,  110,
  110,  110,   24,  760,    0,    0,  108,   61,  110,  115,
  116,  110,  -26,  -26, -118,  471, -117,    0,    0,    0,
    0,    0,    0,  110,  110,  110,  110,  110,  110,  110,
  110,  110,  110,  110,  110,  110,    0,  110,  110,  121,
  482,  103,  504,    0,  110,  131,   70,  760,    9,    0,
    0,  532,  132,    0,  130,  865,  826,   35,   35,  -32,
  -32,   -6,   -6,  -26,  -26,  -26,   35,   35,  554,  760,
  110,   31,  110,   31,  566, -116,    0,  613,  110,    0,
 -101,  110,  110,    0,  135,  133,    0,  670,  -88,   31,
    0,    0,    0,  760,  137,    0,  730,    0,  110,   31,
    0,    0,    0,    0,  138,    0,   31,    0,
};
final static short yyrindex[] = {                         0,
    0,    0,    0,  190,    0,   66,    0,    0,    0,    0,
   66,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,  140,    0,    0,    0,  150,    0,  150,    0,    0,
    0,  159,    0,    0,    0,    0,    0,    0,    0,    0,
    0,  -55,    0,    0,    0,    0,    0,  -53,    0,    0,
    0,    0,    0,    0,    0,  -64,  -64,  -64,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,  815,  444,    0,    0,    0,  -64,
  -55,  -64,   90,  149,    0,    0,    0,    0,  -64,    0,
    0,  -64,  151,  160,    0,    0,    0,    0,    0,    0,
    0,    0,    0,  -64,  -64,  -64,  -64,  -64,  -64,  -64,
  -64,  -64,  -64,  -64,  -64,  -64,    0,  -64,  -64,  123,
    0,    0,    0,    0,  -64,    0,  -64,   64,    0,    0,
    0,    0,    0,    0,    0,    4,    2,  546,  621,    8,
   58,  875,  899,  352,  409,  418,  698,  887,    0,  -22,
  -25,  -55,  -64,  -55,    0,    0,    0,    0,  -64,    0,
    0,  -64,  -64,    0,    0,  174,    0,    0,  -33,  -55,
    0,    0,    0,   84,    0,    0,    0,    0,  -14,  -55,
    0,    0,    0,    0,    0,    0,  -55,    0,
};
final static short yygindex[] = {                         0,
    0,  213,    7,   68,   72,  219,  229,    0,  197,    0,
   23,    0,  189,  -68,    0,    0,    0,    0,    0,    0,
    0,    0, 1070,    0,    0,  -44,  835,    0,    0,    0,
    0,  106,    0,
};
final static int YYTABLESIZE=1243;
static short yytable[];
static { yytable();}
static void yytable(){
yytable = new short[]{                         92,
   33,   33,   92,   47,  124,   94,   92,   85,  181,  122,
  120,   92,  121,  127,  123,   84,    1,    6,   45,  127,
   33,   24,  132,    7,    9,   92,   47,  126,   11,  125,
  124,   67,   12,   29,   24,  122,   45,   13,   68,  127,
  123,    2,   70,   66,   69,   70,   85,   69,   63,  170,
   16,   63,  169,   17,   18,   19,   20,   21,  128,   70,
   70,   69,   69,   67,  128,   63,   63,   49,   31,   51,
   68,  124,   97,   36,   37,   66,  122,  120,   38,  121,
  127,  123,   45,   67,  128,   93,   26,   46,   47,   92,
   68,   92,   48,   30,   70,   66,   69,  108,   64,   26,
   63,   64,   67,   39,   86,   39,   90,   86,   91,   68,
   99,  109,  100,   50,   66,   64,   64,   85,  101,   85,
  195,   48,   43,   69,   85,  128,  102,   85,  107,   43,
  130,   98,  110,  111,   43,   85,   17,   18,   19,   20,
   21,  112,   67,  113,   85,   85,  129,  136,  134,   68,
   64,  137,   85,   48,   66,  140,  141,  143,  145,   50,
  161,  163,   37,   50,   50,   50,   50,   50,   50,   50,
  182,  167,  172,  173,  185,  188,  169,  193,  197,  190,
   50,   50,   50,   50,   50,   32,   35,   72,   15,    1,
   20,   72,   72,   72,   72,   72,   73,   72,    5,   19,
   73,   73,   73,   73,   73,   44,   73,   93,   72,   72,
   72,   49,   72,   50,   83,   50,    8,   73,   73,   73,
   49,   73,   49,   92,   92,   92,   92,   92,   92,   14,
   92,   92,   92,   92,   43,   92,   92,   92,   92,   92,
   92,   92,   92,   72,   28,    0,   92,    0,  116,  117,
   49,   92,   73,   92,   92,   17,   18,   19,   20,   21,
   53,   49,   54,   55,   56,   57,  176,   58,   59,   60,
   61,   62,   63,   64,    0,    0,    0,    0,   65,   70,
   69,   69,    0,   70,   63,   63,   71,   17,   18,   19,
   20,   21,   53,    0,   54,   55,   56,   57,    0,   58,
   59,   60,   61,   62,   63,   64,    0,    0,    0,    0,
   65,  105,   53,    0,   54,   70,    0,    0,   71,    0,
    0,   60,    0,   62,   63,   64,    0,    0,    0,    0,
   65,   53,    0,   54,   64,   64,    0,    0,   71,    0,
   60,    0,   62,   63,   64,    0,    0,    0,    0,   65,
  177,   43,  179,   43,    0,    0,    0,   71,    0,    0,
   43,    0,   43,   43,   43,   43,    0,    0,  192,   43,
    0,   53,    0,   54,    0,    0,    0,   43,  196,    0,
   60,    0,   62,   63,   64,  198,    0,    0,   60,   65,
    0,    0,   60,   60,   60,   60,   60,   71,   60,   50,
   50,    0,    0,   50,   50,   50,   50,    0,    0,   60,
   60,   60,    0,   60,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,   72,   72,    0,
    0,   72,   72,   72,   72,    0,   73,   73,    0,    0,
   73,   73,   73,   73,   60,   61,    0,   23,    0,   61,
   61,   61,   61,   61,   62,   61,    0,    0,   62,   62,
   62,   62,   62,    0,   62,    0,   61,   61,   61,    0,
   61,    0,    0,    0,    0,   62,   62,   62,    0,   62,
   56,    0,    0,    0,   46,   56,   56,    0,   56,   56,
   56,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,   61,   46,   56,    0,   56,    0,  124,    0,    0,
   62,  144,  122,  120,    0,  121,  127,  123,  124,    0,
    0,    0,  162,  122,  120,    0,  121,  127,  123,    0,
  126,    0,  125,    0,   56,    0,    0,    0,    0,    0,
  124,  126,    0,  125,  164,  122,  120,    0,  121,  127,
  123,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,  128,    0,  126,    0,  125,    0,    0,  124,    0,
    0,    0,  128,  122,  120,  171,  121,  127,  123,   17,
   18,   19,   20,   21,    0,    0,   67,    0,    0,   67,
  124,  126,    0,  125,  128,  122,  120,    0,  121,  127,
  123,   22,  124,   67,   67,    0,    0,  122,  120,    0,
  121,  127,  123,  126,    0,  125,    0,    0,    0,    0,
    0,    0,  128,  180,    0,  126,    0,  125,   60,   60,
    0,    0,   60,   60,   60,   60,    0,    0,   67,    0,
    0,    0,    0,    0,  128,    0,  174,    0,    0,  124,
    0,    0,    0,    0,  122,  120,  128,  121,  127,  123,
    0,   68,    0,    0,   68,    0,    0,    0,    0,    0,
    0,    0,  126,    0,  125,    0,    0,    0,   68,   68,
    0,    0,    0,    0,    0,   61,   61,    0,    0,   61,
   61,   61,   61,    0,   62,   62,    0,    0,   62,   62,
   62,   62,    0,  128,    0,  183,  124,    0,    0,    0,
    0,  122,  120,   68,  121,  127,  123,    0,    0,    0,
   56,   56,    0,    0,   56,   56,   56,   56,  189,  126,
    0,  125,    0,    0,    0,    0,    0,    0,   66,    0,
    0,   66,    0,    0,    0,    0,   34,  114,  115,    0,
    0,  116,  117,  118,  119,   66,   66,    0,  114,  115,
  128,    0,  116,  117,  118,  119,  124,    0,    0,    0,
  194,  122,  120,    0,  121,  127,  123,    0,    0,    0,
  114,  115,    0,    0,  116,  117,  118,  119,    0,  126,
   66,  125,    0,    0,    0,    0,  124,    0,    0,    0,
    0,  122,  120,    0,  121,  127,  123,    0,  114,  115,
    0,    0,  116,  117,  118,  119,    0,    0,    0,  126,
  128,  125,   67,   67,    0,    0,    0,    0,   67,   67,
  114,  115,    0,    0,  116,  117,  118,  119,    0,    0,
    0,    0,  114,  115,    0,    0,  116,  117,  118,  119,
  128,   55,    0,    0,    0,    0,   55,   55,    0,   55,
   55,   55,  124,    0,    0,    0,    0,  122,  120,    0,
  121,  127,  123,    0,   55,    0,   55,    0,   17,   18,
   19,   20,   21,    0,    0,  126,   86,  125,    0,  114,
  115,    0,    0,  116,  117,  118,  119,   68,   68,    0,
   22,  124,    0,   68,   68,   55,  122,  120,    0,  121,
  127,  123,    0,    0,    0,   58,  128,   58,   58,   58,
    0,    0,    0,    0,  126,   86,  125,   65,    0,    0,
   65,    0,   58,   58,   58,    0,   58,    0,    0,   59,
    0,   59,   59,   59,   65,   65,  114,  115,    0,    0,
  116,  117,  118,  119,    0,  128,   59,   59,   59,    0,
   59,    0,    0,    0,    0,    0,    0,   58,    0,    0,
    0,    0,    0,    0,   66,   66,    0,    0,    0,   65,
   66,   66,    0,    0,    0,    0,    0,    0,    0,    0,
    0,   59,    0,    0,    0,    0,   86,    0,   86,    0,
    0,    0,    0,    0,    0,    0,  114,  115,    0,    0,
  116,  117,  118,  119,   86,    0,    0,    0,    0,    0,
    0,    0,    0,   86,   86,    0,    0,    0,    0,    0,
    0,   86,    0,    0,    0,    0,  114,  115,    0,    0,
  116,  117,  118,  119,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,   55,   55,    0,    0,   55,   55,   55,   55,    0,
    0,    0,  114,    0,    0,    0,  116,  117,  118,  119,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,   94,    0,    0,
    0,    0,    0,    0,    0,  103,  104,  106,    0,    0,
    0,    0,    0,    0,    0,  116,  117,  118,  119,    0,
    0,   58,   58,    0,    0,   58,   58,   58,   58,  131,
    0,  133,    0,   65,   65,    0,    0,    0,  138,   65,
   65,  142,    0,    0,    0,   59,   59,    0,    0,   59,
   59,   59,   59,  146,  147,  148,  149,  150,  151,  152,
  153,  154,  155,  156,  157,  158,    0,  159,  160,    0,
    0,    0,    0,    0,  165,    0,  168,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
  138,    0,  178,    0,    0,    0,    0,    0,  184,    0,
    0,  186,  187,
};
}
static short yycheck[];
static { yycheck(); }
static void yycheck() {
yycheck = new short[] {                         33,
   91,   91,   40,   59,   37,   59,   40,   52,  125,   42,
   43,   45,   45,   46,   47,   41,  261,  276,   41,   46,
   91,   15,   91,  261,  263,   59,   41,   60,  276,   62,
   37,   33,  276,  276,   28,   42,   59,  123,   40,   46,
   47,  286,   41,   45,   41,   44,   91,   44,   41,   41,
  123,   44,   44,  257,  258,  259,  260,  261,   91,   58,
   59,   58,   59,   33,   91,   58,   59,   45,   59,   47,
   40,   37,  276,   40,   93,   45,   42,   43,   40,   45,
   46,   47,   41,   33,   91,  123,   15,   44,   41,  123,
   40,  125,  123,   22,   93,   45,   93,  276,   41,   28,
   93,   44,   33,   36,   41,   38,   40,   44,   40,   40,
   40,   59,   40,   46,   45,   58,   59,  162,   40,  164,
  189,  123,   33,  125,   41,   91,   40,   44,   40,   40,
  276,   60,   59,   59,   45,  180,  257,  258,  259,  260,
  261,   59,   33,   59,  189,  190,   61,   40,  125,   40,
   93,   91,  197,  123,   45,   41,   41,  276,  276,   37,
   40,   59,   93,   41,   42,   43,   44,   45,   46,   47,
  287,   41,   41,   44,  276,   41,   44,   41,   41,  268,
   58,   59,   60,   61,   62,  276,  276,   37,  123,    0,
   41,   41,   42,   43,   44,   45,   37,   47,   59,   41,
   41,   42,   43,   44,   45,  276,   47,   59,   58,   59,
   60,  276,   62,   91,   41,   93,    4,   58,   59,   60,
  276,   62,  276,  257,  258,  259,  260,  261,  262,   11,
  264,  265,  266,  267,   38,  269,  270,  271,  272,  273,
  274,  275,  276,   93,   16,   -1,  280,   -1,  281,  282,
  276,  285,   93,  287,  288,  257,  258,  259,  260,  261,
  262,  276,  264,  265,  266,  267,  161,  269,  270,  271,
  272,  273,  274,  275,   -1,   -1,   -1,   -1,  280,  278,
  277,  278,   -1,  285,  277,  278,  288,  257,  258,  259,
  260,  261,  262,   -1,  264,  265,  266,  267,   -1,  269,
  270,  271,  272,  273,  274,  275,   -1,   -1,   -1,   -1,
  280,  261,  262,   -1,  264,  285,   -1,   -1,  288,   -1,
   -1,  271,   -1,  273,  274,  275,   -1,   -1,   -1,   -1,
  280,  262,   -1,  264,  277,  278,   -1,   -1,  288,   -1,
  271,   -1,  273,  274,  275,   -1,   -1,   -1,   -1,  280,
  162,  262,  164,  264,   -1,   -1,   -1,  288,   -1,   -1,
  271,   -1,  273,  274,  275,  276,   -1,   -1,  180,  280,
   -1,  262,   -1,  264,   -1,   -1,   -1,  288,  190,   -1,
  271,   -1,  273,  274,  275,  197,   -1,   -1,   37,  280,
   -1,   -1,   41,   42,   43,   44,   45,  288,   47,  277,
  278,   -1,   -1,  281,  282,  283,  284,   -1,   -1,   58,
   59,   60,   -1,   62,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,  277,  278,   -1,
   -1,  281,  282,  283,  284,   -1,  277,  278,   -1,   -1,
  281,  282,  283,  284,   93,   37,   -1,  125,   -1,   41,
   42,   43,   44,   45,   37,   47,   -1,   -1,   41,   42,
   43,   44,   45,   -1,   47,   -1,   58,   59,   60,   -1,
   62,   -1,   -1,   -1,   -1,   58,   59,   60,   -1,   62,
   37,   -1,   -1,   -1,   41,   42,   43,   -1,   45,   46,
   47,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   93,   59,   60,   -1,   62,   -1,   37,   -1,   -1,
   93,   41,   42,   43,   -1,   45,   46,   47,   37,   -1,
   -1,   -1,   41,   42,   43,   -1,   45,   46,   47,   -1,
   60,   -1,   62,   -1,   91,   -1,   -1,   -1,   -1,   -1,
   37,   60,   -1,   62,   41,   42,   43,   -1,   45,   46,
   47,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   91,   -1,   60,   -1,   62,   -1,   -1,   37,   -1,
   -1,   -1,   91,   42,   43,   44,   45,   46,   47,  257,
  258,  259,  260,  261,   -1,   -1,   41,   -1,   -1,   44,
   37,   60,   -1,   62,   91,   42,   43,   -1,   45,   46,
   47,  279,   37,   58,   59,   -1,   -1,   42,   43,   -1,
   45,   46,   47,   60,   -1,   62,   -1,   -1,   -1,   -1,
   -1,   -1,   91,   58,   -1,   60,   -1,   62,  277,  278,
   -1,   -1,  281,  282,  283,  284,   -1,   -1,   93,   -1,
   -1,   -1,   -1,   -1,   91,   -1,   93,   -1,   -1,   37,
   -1,   -1,   -1,   -1,   42,   43,   91,   45,   46,   47,
   -1,   41,   -1,   -1,   44,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   60,   -1,   62,   -1,   -1,   -1,   58,   59,
   -1,   -1,   -1,   -1,   -1,  277,  278,   -1,   -1,  281,
  282,  283,  284,   -1,  277,  278,   -1,   -1,  281,  282,
  283,  284,   -1,   91,   -1,   93,   37,   -1,   -1,   -1,
   -1,   42,   43,   93,   45,   46,   47,   -1,   -1,   -1,
  277,  278,   -1,   -1,  281,  282,  283,  284,   59,   60,
   -1,   62,   -1,   -1,   -1,   -1,   -1,   -1,   41,   -1,
   -1,   44,   -1,   -1,   -1,   -1,  125,  277,  278,   -1,
   -1,  281,  282,  283,  284,   58,   59,   -1,  277,  278,
   91,   -1,  281,  282,  283,  284,   37,   -1,   -1,   -1,
   41,   42,   43,   -1,   45,   46,   47,   -1,   -1,   -1,
  277,  278,   -1,   -1,  281,  282,  283,  284,   -1,   60,
   93,   62,   -1,   -1,   -1,   -1,   37,   -1,   -1,   -1,
   -1,   42,   43,   -1,   45,   46,   47,   -1,  277,  278,
   -1,   -1,  281,  282,  283,  284,   -1,   -1,   -1,   60,
   91,   62,  277,  278,   -1,   -1,   -1,   -1,  283,  284,
  277,  278,   -1,   -1,  281,  282,  283,  284,   -1,   -1,
   -1,   -1,  277,  278,   -1,   -1,  281,  282,  283,  284,
   91,   37,   -1,   -1,   -1,   -1,   42,   43,   -1,   45,
   46,   47,   37,   -1,   -1,   -1,   -1,   42,   43,   -1,
   45,   46,   47,   -1,   60,   -1,   62,   -1,  257,  258,
  259,  260,  261,   -1,   -1,   60,   52,   62,   -1,  277,
  278,   -1,   -1,  281,  282,  283,  284,  277,  278,   -1,
  279,   37,   -1,  283,  284,   91,   42,   43,   -1,   45,
   46,   47,   -1,   -1,   -1,   41,   91,   43,   44,   45,
   -1,   -1,   -1,   -1,   60,   91,   62,   41,   -1,   -1,
   44,   -1,   58,   59,   60,   -1,   62,   -1,   -1,   41,
   -1,   43,   44,   45,   58,   59,  277,  278,   -1,   -1,
  281,  282,  283,  284,   -1,   91,   58,   59,   60,   -1,
   62,   -1,   -1,   -1,   -1,   -1,   -1,   93,   -1,   -1,
   -1,   -1,   -1,   -1,  277,  278,   -1,   -1,   -1,   93,
  283,  284,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   93,   -1,   -1,   -1,   -1,  162,   -1,  164,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,  277,  278,   -1,   -1,
  281,  282,  283,  284,  180,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,  189,  190,   -1,   -1,   -1,   -1,   -1,
   -1,  197,   -1,   -1,   -1,   -1,  277,  278,   -1,   -1,
  281,  282,  283,  284,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,  277,  278,   -1,   -1,  281,  282,  283,  284,   -1,
   -1,   -1,  277,   -1,   -1,   -1,  281,  282,  283,  284,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   58,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   66,   67,   68,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,  281,  282,  283,  284,   -1,
   -1,  277,  278,   -1,   -1,  281,  282,  283,  284,   90,
   -1,   92,   -1,  277,  278,   -1,   -1,   -1,   99,  283,
  284,  102,   -1,   -1,   -1,  277,  278,   -1,   -1,  281,
  282,  283,  284,  114,  115,  116,  117,  118,  119,  120,
  121,  122,  123,  124,  125,  126,   -1,  128,  129,   -1,
   -1,   -1,   -1,   -1,  135,   -1,  137,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
  161,   -1,  163,   -1,   -1,   -1,   -1,   -1,  169,   -1,
   -1,  172,  173,
};
}
final static short YYFINAL=3;
final static short YYMAXTOKEN=290;
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
"SEALED","SEP","VAR","UMINUS","EMPTY",
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

//#line 471 "Parser.y"
    
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
//#line 651 "Parser.java"
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
case 39:
//#line 208 "Parser.y"
{
						yyval.stmt = new Tree.Scopy(val_peek(1).expr, val_peek(3).ident, val_peek(5).loc);
					}
break;
case 40:
//#line 214 "Parser.y"
{
                        val_peek(2).slist.add(val_peek(1).stmt);
                        yyval.stmt = new Tree.GuardStmt(val_peek(2).slist, val_peek(4).loc);
                    }
break;
case 41:
//#line 219 "Parser.y"
{
				        yyval.stmt = new Tree.GuardStmt(null, val_peek(2).loc);
				    }
break;
case 42:
//#line 225 "Parser.y"
{
                       yyval.slist.add(val_peek(1).stmt);
                    }
break;
case 43:
//#line 229 "Parser.y"
{
				    	yyval = new SemValue();
				    	yyval.slist = new ArrayList<Tree>();
				    }
break;
case 44:
//#line 236 "Parser.y"
{
						yyval.stmt = new Tree.Guard(val_peek(2).expr, val_peek(0).stmt, val_peek(2).loc);
					}
break;
case 45:
//#line 242 "Parser.y"
{
						yyval.stmt = new Tree.Assign(val_peek(2).lvalue, val_peek(0).expr, val_peek(1).loc);
					}
break;
case 46:
//#line 246 "Parser.y"
{
                		yyval.stmt = new Tree.Exec(val_peek(0).expr, val_peek(0).loc);
                	}
break;
case 47:
//#line 250 "Parser.y"
{
                		yyval = new SemValue();
                	}
break;
case 49:
//#line 257 "Parser.y"
{
                		yyval = new SemValue();
                	}
break;
case 50:
//#line 263 "Parser.y"
{
						yyval.lvalue = new Tree.Ident(false, val_peek(1).expr, val_peek(0).ident, val_peek(0).loc);
						if (val_peek(1).loc == null) {
							yyval.loc = val_peek(0).loc;
						}
					}
break;
case 51:
//#line 270 "Parser.y"
{
                		yyval.lvalue = new Tree.Indexed(val_peek(3).expr, val_peek(1).expr, val_peek(3).loc);
                	}
break;
case 53:
//#line 277 "Parser.y"
{
                    	yyval.lvalue = new Tree.Ident(true, null, val_peek(0).ident, val_peek(0).loc);
                    }
break;
case 54:
//#line 283 "Parser.y"
{
						yyval.expr = new Tree.CallExpr(val_peek(4).expr, val_peek(3).ident, val_peek(1).elist, val_peek(3).loc);
						if (val_peek(4).loc == null) {
							yyval.loc = val_peek(3).loc;
						}
					}
break;
case 55:
//#line 292 "Parser.y"
{
						yyval.expr = val_peek(0).lvalue;
					}
break;
case 58:
//#line 298 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.PLUS, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 59:
//#line 302 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.MINUS, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 60:
//#line 306 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.MUL, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 61:
//#line 310 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.DIV, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 62:
//#line 314 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.MOD, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 63:
//#line 318 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.EQ, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 64:
//#line 322 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.NE, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 65:
//#line 326 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.LT, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 66:
//#line 330 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.GT, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 67:
//#line 334 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.LE, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 68:
//#line 338 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.GE, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 69:
//#line 342 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.AND, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 70:
//#line 346 "Parser.y"
{
                		yyval.expr = new Tree.Binary(Tree.OR, val_peek(2).expr, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 71:
//#line 350 "Parser.y"
{
                		yyval = val_peek(1);
                	}
break;
case 72:
//#line 354 "Parser.y"
{
                		yyval.expr = new Tree.Unary(Tree.NEG, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 73:
//#line 358 "Parser.y"
{
                		yyval.expr = new Tree.Unary(Tree.NOT, val_peek(0).expr, val_peek(1).loc);
                	}
break;
case 74:
//#line 362 "Parser.y"
{
                		yyval.expr = new Tree.ReadIntExpr(val_peek(2).loc);
                	}
break;
case 75:
//#line 366 "Parser.y"
{
                		yyval.expr = new Tree.ReadLineExpr(val_peek(2).loc);
                	}
break;
case 76:
//#line 370 "Parser.y"
{
                		yyval.expr = new Tree.ThisExpr(val_peek(0).loc);
                	}
break;
case 77:
//#line 374 "Parser.y"
{
                		yyval.expr = new Tree.NewClass(val_peek(2).ident, val_peek(3).loc);
                	}
break;
case 78:
//#line 378 "Parser.y"
{
                		yyval.expr = new Tree.NewArray(val_peek(3).type, val_peek(1).expr, val_peek(4).loc);
                	}
break;
case 79:
//#line 382 "Parser.y"
{
                		yyval.expr = new Tree.TypeTest(val_peek(3).expr, val_peek(1).ident, val_peek(5).loc);
                	}
break;
case 80:
//#line 386 "Parser.y"
{
                		yyval.expr = new Tree.TypeCast(val_peek(2).ident, val_peek(0).expr, val_peek(0).loc);
                	}
break;
case 81:
//#line 392 "Parser.y"
{
						yyval.expr = new Tree.Literal(val_peek(0).typeTag, val_peek(0).literal, val_peek(0).loc);
					}
break;
case 82:
//#line 396 "Parser.y"
{
						yyval.expr = new Null(val_peek(0).loc);
					}
break;
case 84:
//#line 403 "Parser.y"
{
                		yyval = new SemValue();
                		yyval.elist = new ArrayList<Tree.Expr>();
                	}
break;
case 85:
//#line 410 "Parser.y"
{
						yyval.elist.add(val_peek(0).expr);
					}
break;
case 86:
//#line 414 "Parser.y"
{
                		yyval.elist = new ArrayList<Tree.Expr>();
						yyval.elist.add(val_peek(0).expr);
                	}
break;
case 87:
//#line 421 "Parser.y"
{
						yyval.stmt = new Tree.WhileLoop(val_peek(2).expr, val_peek(0).stmt, val_peek(4).loc);
					}
break;
case 88:
//#line 427 "Parser.y"
{
						yyval.stmt = new Tree.ForLoop(val_peek(6).stmt, val_peek(4).expr, val_peek(2).stmt, val_peek(0).stmt, val_peek(8).loc);
					}
break;
case 89:
//#line 433 "Parser.y"
{
						yyval.stmt = new Tree.Break(val_peek(0).loc);
					}
break;
case 90:
//#line 439 "Parser.y"
{
						yyval.stmt = new Tree.If(val_peek(3).expr, val_peek(1).stmt, val_peek(0).stmt, val_peek(5).loc);
					}
break;
case 91:
//#line 445 "Parser.y"
{
						yyval.stmt = val_peek(0).stmt;
					}
break;
case 92:
//#line 449 "Parser.y"
{
						yyval = new SemValue();
					}
break;
case 93:
//#line 455 "Parser.y"
{
						yyval.stmt = new Tree.Return(val_peek(0).expr, val_peek(1).loc);
					}
break;
case 94:
//#line 459 "Parser.y"
{
                		yyval.stmt = new Tree.Return(null, val_peek(0).loc);
                	}
break;
case 95:
//#line 465 "Parser.y"
{
						yyval.stmt = new Print(val_peek(1).elist, val_peek(3).loc);
					}
break;
//#line 1288 "Parser.java"
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
