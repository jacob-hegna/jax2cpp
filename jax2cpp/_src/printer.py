from jax2cpp._src import ast


class Printer:
    def __init__(self, spaces=2):
        self._spaces = spaces

    def _indent(self, level) -> str:
        return " " * self._spaces * level

    def _print_return(self, s: ast.ReturnStmt, level: int) -> str:
        expr = self.print(s.expr, level=0)
        return f"{self._indent(level)}return {expr};"

    def _print_expr_stmt(self, s: ast.ExprStmt, level: int) -> str:
        return f"{self._indent(level)}{self.print(s.expr, level=level)};"

    def _print_assignment(self, e: ast.AssignmentExpr, level: int) -> str:
        lhs = self.print(e.expr, level=level)
        return f"{e.typename} {e.name} = {lhs}"

    def _print_call(self, c: ast.CallExpr, level: int) -> str:
        argstr = ", ".join([self.print(arg, level=level) for arg in c.args])
        return f"{c.name}({argstr})"

    def _print_binary(self, e: ast.BinaryOpExpr, level: int) -> str:
        lhs = self.print(e.lhs, level=level)
        rhs = self.print(e.rhs, level=level)
        return f"{lhs} {e.op} {rhs}"

    def _print_parameter(self, p: ast.Parameter, level: int) -> str:
        del level
        return " ".join([p.typename, p.name])

    def _print_function(self, f: ast.Function, level: int) -> str:
        paramstr = ", ".join([self.print(p, level=0) for p in f.parameters])
        typename_space = " " if f.typename else ""
        modifiers = (" ".join(f.modifiers) + " ") if f.modifiers else ""
        return (
            f"{self._indent(level)}{f.typename}{typename_space}{f.name}({paramstr}) {modifiers}{{\n"
            + "\n".join([self.print(s, level=level + 1) for s in f.body])
            + f"\n{self._indent(level)}}}"
        )

    def _print_directive(self, d: ast.Directive, level: int) -> str:
        return f"{self._indent(level)}#{d.kind} {d.value}"

    def _print_class(self, c: ast.ClassDecl, level: int) -> str:
        return (
            (f"template <{c.template}>\n" if c.template else "")
            + f"class {c.name} {{\n"
            + "public: \n"
            + "\n".join([self.print(t, level=level + 1) for t in c.types])
            + "\n"
            + (self.print(c.constructor, level=level + 1) if c.constructor else "")
            + "\n"
            + "\n".join([self.print(m, level=level + 1) for m in c.methods])
            + "\n"
            + "\n".join([self.print(m, level=level + 1) for m in c.members])
            + "\n};"
        )

    def _print_namespace(self, n: ast.Namespace, level: int) -> str:
        return (
            f"namespace {n.name} {{\n"
            + "\n".join([self.print(d, level=level) for d in n.decls])
            + "\n}"
        )

    def _print_module(self, m: ast.Module, level: int) -> str:
        return "\n".join(
            [self.print(d, level=level) for d in m.directives]
            + [""]
            + [self.print(d, level=level) for d in m.decls]
        )

    def print(self, n: ast.Node, *, level: int) -> str:
        match n:
            case ast.Module():
                return self._print_module(n, level)
            case ast.Namespace():
                return self._print_namespace(n, level)
            case ast.ClassDecl():
                return self._print_class(n, level)
            case ast.Directive():
                return self._print_directive(n, level)
            case ast.Function():
                return self._print_function(n, level)
            case ast.Parameter():
                return self._print_parameter(n, level)
            case ast.VarExpr():
                return n.name
            case ast.LiteralExpr():
                return n.literal
            case ast.BinaryOpExpr():
                return self._print_binary(n, level)
            case ast.CallExpr():
                return self._print_call(n, level)
            case ast.AssignmentExpr():
                return self._print_assignment(n, level)
            case ast.ExprStmt():
                return self._print_expr_stmt(n, level)
            case ast.ReturnStmt():
                return self._print_return(n, level)
            case _:
                raise RuntimeError(f"Unhandled type for Printer.print: {type(n)}")


def print_ast(n: ast.Node) -> str:
    p = Printer()
    return p.print(n, level=0)
