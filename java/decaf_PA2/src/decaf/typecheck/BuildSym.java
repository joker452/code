package decaf.typecheck;

import java.util.Iterator;

import decaf.Driver;
import decaf.tree.Tree;
import decaf.error.BadArrElementError;
import decaf.error.BadInheritanceError;
import decaf.error.BadOverrideError;
import decaf.error.BadSealedInherError;
import decaf.error.BadVarTypeError;
import decaf.error.ClassNotFoundError;
import decaf.error.DecafError;
import decaf.error.DeclConflictError;
import decaf.error.NoMainClassError;
import decaf.error.OverridingVarError;
import decaf.scope.ClassScope;
import decaf.scope.GlobalScope;
import decaf.scope.LocalScope;
import decaf.scope.ScopeStack;
import decaf.symbol.Class;
import decaf.symbol.Function;
import decaf.symbol.Symbol;
import decaf.symbol.Variable;
import decaf.type.BaseType;
import decaf.type.FuncType;

public class BuildSym extends Tree.Visitor {

	private ScopeStack table;

	private void issueError(DecafError error) {
		Driver.getDriver().issueError(error);
	}

	public BuildSym(ScopeStack table) {
		this.table = table;
	}

	public static void buildSymbol(Tree.TopLevel tree) {
		new BuildSym(Driver.getDriver().getTable()).visitTopLevel(tree);
	}

	// root
	@Override
	public void visitTopLevel(Tree.TopLevel program) {
		program.globalScope = new GlobalScope();
		table.open(program.globalScope);
		// all the operations in this for loop is in GlobalScope
		// this aim
		for (Tree.ClassDef cd : program.classes) {
			Class c = new Class(cd.name, cd.parent, cd.getLocation());
			Class earlier = table.lookupClass(cd.name);
			if (earlier != null) {
				issueError(new DeclConflictError(cd.getLocation(), cd.name,
						earlier.getLocation()));
			} else {
				// put the class symbol in the GlobalScope
				table.declare(c);
			}
			// put the class symbol in the ClassDef node
			cd.symbol = c;
		}

		// check inheritance for parent class
		for (Tree.ClassDef cd : program.classes) {
			Class c = cd.symbol;
			if (cd.parent != null) {
				// parent class not declared
				if (c.getParent() == null) {
					issueError(new ClassNotFoundError(cd.getLocation(), cd.parent));
					c.dettachParent();
				}
				else
				{
					// sealed parent class
					for (Tree.ClassDef cls : program.classes)
						if (cls.name.equals(cd.parent))
							if (cls.isSealed == true) {
								issueError(new BadSealedInherError(cd.getLocation()));
								c.dettachParent();
							}
				}
			}	
			// cyclical inheritance
			if (calcOrder(c) <= calcOrder(c.getParent())) {
				issueError(new BadInheritanceError(cd.getLocation()));
				c.dettachParent();
			}
		}

		// create type for each class symbol
		for (Tree.ClassDef cd : program.classes) {
			cd.symbol.createType();
		}
		
		// set the symbol corresponds to class Main
		for (Tree.ClassDef cd : program.classes) {
			cd.accept(this);
			if (Driver.getDriver().getOption().getMainClassName().equals(
					cd.name)) {
				program.main = cd.symbol;
			}
		}

		for (Tree.ClassDef cd : program.classes) {
			checkOverride(cd.symbol);
		}

		if (!isMainClass(program.main)) {
			issueError(new NoMainClassError(Driver.getDriver().getOption()
					.getMainClassName()));
		}
		table.close();
	}

	// visiting declarations
	@Override
	public void visitClassDef(Tree.ClassDef classDef) {
		table.open(classDef.symbol.getAssociatedScope());
		for (Tree f : classDef.fields) {
			f.accept(this);
		}
		table.close();
	}

	@Override
	public void visitVarDef(Tree.VarDef varDef) {
		varDef.type.accept(this);
		if (varDef.type.type.equal(BaseType.VOID)) {
			issueError(new BadVarTypeError(varDef.getLocation(), varDef.name));
			// for argList
			varDef.symbol = new Variable(".error", BaseType.ERROR, varDef
					.getLocation());
			return;
		}
		Variable v = new Variable(varDef.name, varDef.type.type, 
				varDef.getLocation());
		// look for namesake variable
		Symbol sym = table.lookup(varDef.name, true);
		if (sym != null) {
			// namesake variable in one scope is not allowed
			if (table.getCurrentScope().equals(sym.getScope())) {
				issueError(new DeclConflictError(v.getLocation(), v.getName(),
						sym.getLocation()));
			} 
			// neither is between a formal scope and a local scope
			else if ((sym.getScope().isFormalScope() && table.getCurrentScope().isLocalScope() && ((LocalScope)table.getCurrentScope()).isCombinedtoFormal() )) {
				issueError(new DeclConflictError(v.getLocation(), v.getName(),
						sym.getLocation()));
			} else {
				table.declare(v);
			}
		} else {
			table.declare(v);
		}
		varDef.symbol = v;
	}

	@Override
	public void visitMethodDef(Tree.MethodDef funcDef) {
		// check returnType
		funcDef.returnType.accept(this);
		Function f = new Function(funcDef.statik, funcDef.name,
				funcDef.returnType.type, funcDef.body, funcDef.getLocation());
		funcDef.symbol = f;
		// check function namesake in the same ClassScope
		Symbol sym = table.lookup(funcDef.name, false);
		if (sym != null) {
			issueError(new DeclConflictError(funcDef.getLocation(),
					funcDef.name, sym.getLocation()));
		} else {
			table.declare(f);
		}
		// open the FormalScope of this function
		table.open(f.getAssociatedScope());
		for (Tree.VarDef d : funcDef.formals) {
			d.accept(this);
			// add the argument's type to the argument type list
			f.appendParam(d.symbol);
		}

		funcDef.body.associatedScope = new LocalScope(funcDef.body);
		funcDef.body.associatedScope.setCombinedtoFormal(true);
		// open the LocalScope of this function
		table.open(funcDef.body.associatedScope);
		for (Tree s : funcDef.body.block) {
			s.accept(this);
		}
		// close the LocalScope
		table.close();
		// close the FormalScope
		table.close();
	}

	// visiting types
	@Override
	public void visitTypeIdent(Tree.TypeIdent type) {
		switch (type.typeTag) {
		case Tree.VOID:
			type.type = BaseType.VOID;
			break;
		case Tree.INT:
			type.type = BaseType.INT;
			break;
		case Tree.BOOL:
			type.type = BaseType.BOOL;
			break;
		default:
			type.type = BaseType.STRING;
		}
	}

	@Override
	public void visitTypeClass(Tree.TypeClass typeClass) {
		Class c = table.lookupClass(typeClass.name);
		if (c == null) {
			issueError(new ClassNotFoundError(typeClass.getLocation(),
					typeClass.name));
			typeClass.type = BaseType.ERROR;
		} else {
			typeClass.type = c.getType();
		}
	}

	@Override
	public void visitTypeArray(Tree.TypeArray typeArray) {
		typeArray.elementType.accept(this);
		if (typeArray.elementType.type.equal(BaseType.ERROR)) {
			typeArray.type = BaseType.ERROR;
		} else if (typeArray.elementType.type.equal(BaseType.VOID)) {
			issueError(new BadArrElementError(typeArray.getLocation()));
			typeArray.type = BaseType.ERROR;
		} else {
			typeArray.type = new decaf.type.ArrayType(
					typeArray.elementType.type);
		}
	}

	// for VarDecl in LocalScope
	@Override
	public void visitBlock(Tree.Block block) {
		block.associatedScope = new LocalScope(block);
		table.open(block.associatedScope);
		for (Tree s : block.block) {
			s.accept(this);
		}
		table.close();
	}

	@Override
	public void visitForLoop(Tree.ForLoop forLoop) {
		if (forLoop.loopBody != null) {
			forLoop.loopBody.accept(this);
		}
	}

	@Override
	public void visitIf(Tree.If ifStmt) {
		if (ifStmt.trueBranch != null) {
			ifStmt.trueBranch.accept(this);
		}
		if (ifStmt.falseBranch != null) {
			ifStmt.falseBranch.accept(this);
		}
	}

	@Override
	public void visitWhileLoop(Tree.WhileLoop whileLoop) {
		if (whileLoop.loopBody != null) {
			whileLoop.loopBody.accept(this);
		}
	}
	
	@Override
	public void visitGuardStmt(Tree.GuardStmt guardStmt) {
		if (guardStmt.guard != null)
			for (Tree guard: guardStmt.guard) {
				guard.accept(this);
			}
	}
	
	@Override
	public void visitGuard(Tree.Guard guard) {
		if (guard.stmt != null)
			guard.stmt.accept(this);	
	}
	/**
	 * subclass's order is parent class's order + 1
	 * the upper most class has order 0
	 */
	private int calcOrder(Class c) {
		if (c == null) {
			return -1;
		}
		if (c.getOrder() < 0) {
			c.setOrder(0);
			c.setOrder(calcOrder(c.getParent()) + 1);
		}
		return c.getOrder();
	}

	private void checkOverride(Class c) {
		// check each symbol only once
		if (c.isCheck()) {
			return;
		}
		Class parent = c.getParent();
		if (parent == null) {
			return;
		}
		
		// check parent class first
		checkOverride(parent);

		ClassScope parentScope = parent.getAssociatedScope();
		ClassScope subScope = c.getAssociatedScope();
		table.open(parentScope);
		Iterator<Symbol> iter = subScope.iterator();
		while (iter.hasNext()) {
			Symbol suspect = iter.next();
			Symbol sym = table.lookup(suspect.getName(), true);
			if (sym != null && !sym.isClass()) {
				// symbol with same name needs to be of same type
				if ((suspect.isVariable() && sym.isFunction())
						|| (suspect.isFunction() && sym.isVariable())) {
					issueError(new DeclConflictError(suspect.getLocation(),
							suspect.getName(), sym.getLocation()));
					iter.remove();
				} else if (suspect.isFunction()) {
					// static function only has one instance
					if (((Function) suspect).isStatik()
							|| ((Function) sym).isStatik()) {
						issueError(new DeclConflictError(suspect.getLocation(),
								suspect.getName(), sym.getLocation()));
						iter.remove();
					// only non static function with proper type can be overridsden
					} else if (!suspect.getType().compatible(sym.getType())) {
						issueError(new BadOverrideError(suspect.getLocation(),
								suspect.getName(),
								((ClassScope) sym.getScope()).getOwner()
										.getName()));
						iter.remove();
					}
				} else if (suspect.isVariable()) {
					issueError(new OverridingVarError(suspect.getLocation(),
							suspect.getName()));
					iter.remove();
				}
			}
		}
		table.close();
		c.setCheck(true);
	}

	/**
	 * lookup function main in c.associatedScope,
	 * return true when there is a main function and
	 * its signature is legal
	 */
	private boolean isMainClass(Class c) {
		if (c == null) {
			return false;
		}
		table.open(c.getAssociatedScope());
		Symbol main = table.lookup(Driver.getDriver().getOption()
				.getMainFuncName(), false);
		if (main == null || !main.isFunction()) {
			return false;
		}
		((Function) main).setMain(true);
		FuncType type = (FuncType) main.getType();
		return type.getReturnType().equal(BaseType.VOID)
				&& type.numOfParams() == 0 && ((Function) main).isStatik();
	}
}
