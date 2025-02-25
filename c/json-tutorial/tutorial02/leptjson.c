#include "leptjson.h"
#include <errno.h>
#include <math.h>
#include <assert.h>  /* assert() */
#include <stdlib.h>  /* NULL, strtod() */

#define EXPECT(c, ch)       do { assert(*c->json == (ch)); c->json++; } while(0)
#define ISDIGIT(ch)			((ch) >= '0' && (ch) <= '9')
#define ISDIGIT1TO9(ch)     ((ch) >= '1' && (ch) <= '9')
typedef struct {
    const char* json;
}lept_context;

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

    v->n = strtod(c->json, NULL);
    if (errno == ERANGE)
    	if (v->n == HUGE_VAL || v->n == -HUGE_VAL)
    		return LEPT_PARSE_NUMBER_TOO_BIG;
    v->type = LEPT_NUMBER;
    c->json = p;
    return LEPT_PARSE_OK;
}

static int lept_parse_value(lept_context* c, lept_value* v) {
    switch (*c->json) {
        case 't':
        case 'f':
        case 'n':  return lept_parse_literal(c, v);
        default:   return lept_parse_number(c, v);
        case '\0': return LEPT_PARSE_EXPECT_VALUE;
    }
}

int lept_parse(lept_value* v, const char* json) {
	 lept_context c;
	    assert(v != NULL);
	    assert(json != NULL);
	    c.json = json;
	    v->type = LEPT_NULL;
	    lept_parse_whitespace(&c);
	    int lept_parse_result = lept_parse_value(&c, v);
	    lept_parse_whitespace(&c);

	    if (lept_parse_result == LEPT_PARSE_OK && c.json[0] != '\0')
	    {
	    	v->type = LEPT_NULL;
	    	return LEPT_PARSE_ROOT_NOT_SINGULAR;
	    }
	    else
	    	return lept_parse_result;
}

lept_type lept_get_type(const lept_value* v) {
    assert(v != NULL);
    return v->type;
}

double lept_get_number(const lept_value* v) {
    assert(v != NULL && v->type == LEPT_NUMBER);
    return v->n;
}
