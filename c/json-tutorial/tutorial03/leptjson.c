#include "leptjson.h"
#include <assert.h>  /* assert() */
#include <errno.h>   /* errno, ERANGE */
#include <math.h>    /* HUGE_VAL */
#include <stdlib.h>  /* NULL, malloc(), realloc(), free(), strtod() */
#include <string.h>  /* memcpy() */

#ifndef LEPT_PARSE_STACK_INIT_SIZE
#define LEPT_PARSE_STACK_INIT_SIZE 256
#endif

#define EXPECT(c, ch)       do { assert(*c->json == (ch)); c->json++; } while(0)
#define ISDIGIT(ch)         ((ch) >= '0' && (ch) <= '9')
#define ISDIGIT1TO9(ch)     ((ch) >= '1' && (ch) <= '9')
#define PUTC(c, ch)         do { *(char*)lept_context_push(c, sizeof(char)) = (ch); } while(0)

typedef struct {
    const char* json;
    char* stack;
    size_t size, top;
}lept_context;

static void* lept_context_push(lept_context* c, size_t size) {
    void* ret;
    assert(size > 0);
    if (c->top + size >= c->size) {
        if (c->size == 0)
            c->size = LEPT_PARSE_STACK_INIT_SIZE;
        while (c->top + size >= c->size)
            c->size += c->size >> 1;  /* c->size * 1.5 */
        c->stack = (char*)realloc(c->stack, c->size);
    }
    ret = c->stack + c->top;
    c->top += size;
    return ret;
}

static void* lept_context_pop(lept_context* c, size_t size) {
    assert(c->top >= size);
    return c->stack + (c->top -= size);
}

static void lept_parse_whitespace(lept_context* c) {
    const char *p = c->json;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')
        p++;
    c->json = p;
}

static int lept_parse_literal(lept_context* c, lept_value* v) {

	switch(*c->json) {
	case 't':
		if (c->json[1] != 'r' || c->json[2] != 'u' || c->json[3] != 'e')
			return LEPT_PARSE_INVALID_VALUE;
		else {
			c->json += 4;
			v->type = LEPT_TRUE;
		}
		break;
	case 'f':
		if (c->json[1] != 'a' || c->json[2] != 'l' || c->json[3] != 's' || c->json[4] != 'e')
			return LEPT_PARSE_INVALID_VALUE;
		else {
			c->json += 5;
			v->type = LEPT_FALSE;
		}
		break;
	case 'n':
		if (c->json[1] != 'u' || c->json[2] != 'l' || c->json[3] != 'l')
			return LEPT_PARSE_INVALID_VALUE;
		else {
			c->json += 4;
			v->type = LEPT_NULL;
		}
		break;
	default:
		return LEPT_PARSE_INVALID_VALUE;
	}

	return LEPT_PARSE_OK;
}

static int lept_parse_number(lept_context* c, lept_value* v) {
    const char* p = c->json;

    // optional '-'
    if (p[0] == '-')
    	++p;

    // int part
    if (p[0] == '0')
    	++p;
    else if (ISDIGIT1TO9(p[0]))
    {
    	while (ISDIGIT(p[0]))
    		++p;
    }
    else
    	return LEPT_PARSE_INVALID_VALUE;

    // frac part
   if (p[0] == '.')
    {
    	if (!ISDIGIT(p[1]))
    		return LEPT_PARSE_INVALID_VALUE;
    	p += 2;
    	while (ISDIGIT(p[0]))
    		++p;
    }

    // exp part
   if (p[0] == 'E' || p[0] == 'e')
    {
    	++p;

    	// optional '-' or '+'
    	if (p[0] == '-' || p[0] == '+')
    		++p;

    	if (ISDIGIT(p[0]))
    		do {
    			++p;
    		} while (ISDIGIT(p[0]));
    	else
    		return LEPT_PARSE_INVALID_VALUE;
    }

    v->u.n = strtod(c->json, NULL);
    if (errno == ERANGE)
    	if (v->u.n == HUGE_VAL || v->u.n == -HUGE_VAL)
    		return LEPT_PARSE_NUMBER_TOO_BIG;
    v->type = LEPT_NUMBER;
    c->json = p;
    return LEPT_PARSE_OK;
}

static int lept_parse_string(lept_context* c, lept_value* v) {
    size_t head = c->top, len;
    const char* p;
    EXPECT(c, '\"');
    p = c->json;
    for (;;) {
        char ch = *p++;
        switch (ch) {
            case '\"':
                len = c->top - head;
                lept_set_string(v, (const char*)lept_context_pop(c, len), len);
                c->json = p;
                return LEPT_PARSE_OK;
            case '\0':
                c->top = head;
                return LEPT_PARSE_MISS_QUOTATION_MARK;
            case '\\':
            	switch(*p++) {
            		case '\"': PUTC(c, '\"'); break;
                    case '\\': PUTC(c, '\\'); break;
                    case '/':  PUTC(c, '/' ); break;
                    case 'b':  PUTC(c, '\b'); break;
                    case 'f':  PUTC(c, '\f'); break;
                    case 'n':  PUTC(c, '\n'); break;
                    case 'r':  PUTC(c, '\r'); break;
                    case 't':  PUTC(c, '\t'); break;
                    default: c->top = head; return LEPT_PARSE_INVALID_STRING_ESCAPE;
            	}
            	break;
            default:
            	if (0x1 <= ch && ch <= 0x1f)
            	{
            		c->top = head;
            		return LEPT_PARSE_INVALID_STRING_CHAR;
            	}
            	else
            		PUTC(c, ch);
        }
    }
}

static int lept_parse_value(lept_context* c, lept_value* v) {
    switch (*c->json) {
    	case 't':
    	case 'f':
    	case 'n':  return lept_parse_literal(c, v);
        default:   return lept_parse_number(c, v);
        case '"':  return lept_parse_string(c, v);
        case '\0': return LEPT_PARSE_EXPECT_VALUE;
    }
}

int lept_parse(lept_value* v, const char* json) {
    lept_context c;
    assert(v != NULL);
    c.json = json;
    c.stack = NULL;
    c.size = c.top = 0;
    lept_init(v);
    lept_parse_whitespace(&c);
    int lept_parse_result = lept_parse_value(&c, v);
 	lept_parse_whitespace(&c);
 	if (lept_parse_result == LEPT_PARSE_OK && c.json[0] != '\0')
 	{
 	   v->type = LEPT_NULL;
 	    return LEPT_PARSE_ROOT_NOT_SINGULAR;
 	}
 	else
 	{
 		assert(c.top == 0);
 		free(c.stack);
 		return lept_parse_result;
 	}
}

void lept_free(lept_value* v) {
    assert(v != NULL);
    if (v->type == LEPT_STRING)
        free(v->u.s.s);
    v->type = LEPT_NULL;
}

lept_type lept_get_type(const lept_value* v) {
    assert(v != NULL);
    return v->type;
}

int lept_get_boolean(const lept_value* v) {
    assert(v != NULL && (v->type == LEPT_FALSE || v->type == LEPT_TRUE));
    return v->type == LEPT_TRUE;
}

void lept_set_boolean(lept_value* v, int b) {
    assert(v != NULL);
    lept_free(v);
    v->type = b ? LEPT_TRUE: LEPT_FALSE;
}

double lept_get_number(const lept_value* v) {
    assert(v != NULL && v->type == LEPT_NUMBER);
    return v->u.n;
}

void lept_set_number(lept_value* v, double n) {
    assert(v != NULL);
    lept_free(v);
    v->u.n = n;
    v->type = LEPT_NUMBER;
}

const char* lept_get_string(const lept_value* v) {
    assert(v != NULL && v->type == LEPT_STRING);
    return v->u.s.s;
}

size_t lept_get_string_length(const lept_value* v) {
    assert(v != NULL && v->type == LEPT_STRING);
    return v->u.s.len;
}

void lept_set_string(lept_value* v, const char* s, size_t len) {
    assert(v != NULL && (s != NULL || len == 0));
    lept_free(v);
    v->u.s.s = (char*)malloc(len + 1);
    memcpy(v->u.s.s, s, len);
    v->u.s.s[len] = '\0';
    v->u.s.len = len;
    v->type = LEPT_STRING;
}
