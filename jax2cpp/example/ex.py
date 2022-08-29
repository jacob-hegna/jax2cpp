import re
from typing import List

import jax
import jax.numpy as jnp

from jax2cpp import ast
from jax2cpp._src.ast import CallExpr
from jax2cpp._src.printer import print_ast


def jnp_to_cpp_array(x: jnp.ndarray) -> str:
    s = str(x.tolist())
    s = re.sub(r"\[", "{", s)
    s = re.sub(r"\]", "}", s)
    return s


def get_shape(shape) -> str:
    return "Shape<" + ", ".join([str(d) for d in shape]) + ">"


def get_tensor_type(aval: jax.core.AbstractValue) -> str:
    match aval:
        case jax.core.ShapedArray():
            if aval.ndim == 0:
                return "FloatT"
            shapestr = get_shape(aval.shape)
            return f"typename lax::template Tensor<FloatT, {shapestr}>"
    return ""


def build_eval_pure(jaxpr: jax.core.ClosedJaxpr) -> ast.Function:
    body = []
    for eq in jaxpr.eqns:
        # if eq.primitive.name == "add":
        #    print(f"{get_shape(eq.invars[0].aval)} | {get_shape(eq.invars[1].aval)}")
        name = "lax::" + eq.primitive.name
        args = [
            ast.VarExpr(name=f"static_cast<FloatT>({e})")
            if isinstance(e, jax.core.Literal)
            else ast.VarExpr(name=str(e))
            for e in eq.invars
        ]
        if eq.primitive.name == "dot_general":
            dims = eq.params["dimension_numbers"][0]
            a, b = dims
            name = f"lax::template dot_general<{a[0]}, {b[0]}>"
        if eq.primitive.name == "transpose":
            shape = get_shape(eq.params["permutation"])
            name = f"lax::template transpose<{shape}>"
        if eq.primitive.name == "squeeze":
            cur_shape = eq.invars[0].aval.shape
            dims = eq.params["dimensions"]
            shape = get_shape(
                tuple(
                    v for v, i in zip(cur_shape, range(len(cur_shape))) if i not in dims
                )
            )
            name = f"lax::template internal_reshape<{shape}>"
        if eq.primitive.name == "broadcast_in_dim":
            desired_shape = eq.params["shape"]
            shuffle = eq.params["broadcast_dimensions"]
            shuffle = tuple(set(range(len(desired_shape))) - set(shuffle)) + shuffle
            cur_shape = eq.invars[0].aval.shape
            reshape = (
                tuple(1 for _ in range(len(desired_shape) - len(cur_shape))) + cur_shape
            )

            shuffle_reshape = tuple(reshape[i] for i in shuffle)
            broadcast = tuple(q // p for p, q in zip(shuffle_reshape, desired_shape))
            name = f"lax::template broadcast_in_dim<{get_shape(reshape)}, {get_shape(broadcast)}, {get_shape(shuffle)}>"
        if eq.primitive.name == "integer_pow":
            exp = eq.params["y"]
            if isinstance(exp, int):
                name = f"lax::template integer_pow_fast<{exp}>"
            else:
                args.append(ast.LiteralExpr(literal=str(exp)))
            # args.append(ast.VarExpr(name=str(eq.params["dimension_numbers"][0])))
        if eq.primitive.name == "reshape":
            shape = get_shape(eq.params["new_sizes"])
            name = f"lax::template internal_reshape<{shape}>"
        if eq.primitive.name == "reduce_sum":
            shape = get_shape(eq.params["axes"])
            name = f"lax::template reduce_sum<{shape}>"
        body.append(
            ast.ExprStmt(
                ast.AssignmentExpr(
                    typename="auto",
                    name=str(eq.outvars[0]),
                    expr=ast.CallExpr(
                        name=name,
                        args=args,
                    ),
                )
            )
        )
    body.append(ast.ReturnStmt(expr=ast.VarExpr(name=str(jaxpr.jaxpr.outvars[0]))))
    return ast.Function(
        name="eval_pure",
        typename="[[nodiscard, gnu::flatten]] static "
        + get_tensor_type(jaxpr.jaxpr.outvars[0].aval),
        parameters=[
            ast.Parameter(typename=get_tensor_type(p.aval), name=str(p))
            for p in jaxpr.jaxpr.constvars + jaxpr.jaxpr.invars
        ],
        body=body,
    )


def to_cpp(jaxpr: jax.core.ClosedJaxpr, name: str) -> ast.Module:
    c = ast.ClassDecl(
        template="Backend backend, class FloatT = float",
        name=name,
        types=[
            ast.ExprStmt(
                expr=ast.AssignmentExpr(
                    name="lax", typename="using", expr=ast.VarExpr(name="Lax<backend>")
                )
            )
        ],
        constructor=ast.Function(
            name=name,
            typename="",
            parameters=[
                ast.Parameter(name=str(c), typename=get_tensor_type(c.aval))
                for c in jaxpr.jaxpr.constvars
            ],
            body=[
                ast.ExprStmt(
                    expr=ast.AssignmentExpr(
                        name=str(c) + "_", typename="", expr=ast.VarExpr(name=str(c))
                    )
                )
                for c in jaxpr.jaxpr.constvars
            ],
        ),
        members=[
            ast.ExprStmt(
                ast.Parameter(name=str(c) + "_", typename=get_tensor_type(c.aval))
            )
            for c in jaxpr.jaxpr.constvars
        ],
        methods=[
            ast.Function(
                name="get",
                typename=f"[[nodiscard]] static {name}",
                parameters=[],
                body=[
                    ast.ExprStmt(
                        expr=ast.AssignmentExpr(
                            typename="auto",
                            name=str(c),
                            expr=ast.CallExpr(
                                name=f"lax::template make_tensor<FloatT, {get_shape(c.aval.shape)}, "
                                + ("std::initializer_list<" * c.aval.ndim)
                                + "FloatT"
                                + (">" * c.aval.ndim)
                                + ">",
                                args=[ast.LiteralExpr(literal=jnp_to_cpp_array(val))],
                            ),
                        )
                    )
                    for c, val in zip(jaxpr.jaxpr.constvars, jaxpr.consts)
                ]
                + [
                    ast.ReturnStmt(
                        expr=CallExpr(
                            name=name,
                            args=[
                                ast.VarExpr(name=str(c)) for c in jaxpr.jaxpr.constvars
                            ],
                        )
                    )
                ],
            ),
            build_eval_pure(jaxpr),
            ast.Function(
                name="eval",
                typename="[[nodiscard, gnu::flatten]] auto",
                parameters=[
                    ast.Parameter(typename=get_tensor_type(p.aval), name=str(p))
                    for p in jaxpr.jaxpr.invars
                ],
                modifiers=["const"],
                body=[
                    ast.ReturnStmt(
                        expr=ast.CallExpr(
                            name="eval_pure",
                            args=[
                                ast.VarExpr(str(p) + "_") for p in jaxpr.jaxpr.constvars
                            ]
                            + [ast.VarExpr(str(p)) for p in jaxpr.jaxpr.invars],
                        )
                    )
                ],
            ),
        ],
    )

    return ast.Module(
        name="asdf",
        directives=[ast.Directive(kind="include", value="<jax2cpp/jax2cpp.h>")],
        decls=[ast.Namespace(name="j2c", decls=[c])],
    )


import haiku as hk


def model(x):
    from functools import partial

    import jaxtree

    m = hk.Sequential(
        [
            # jaxtree.DTreeModule(in_shape=(2,), out_shape=(1,), depth=1, batch=1024)
            hk.Linear(128),
            jax.nn.gelu,
            hk.Linear(64),
            jax.nn.gelu,
            hk.Linear(16),
            jax.nn.gelu,
            hk.Linear(1),
        ]
    )
    return m(x)


def main():
    def f(x, y):
        return (jnp.abs(x) + jnp.array([[1.0, 3.0], [6.0, 9.0]])) ** 2.0

    from functools import partial

    """
    import jaxtree

    """
    jaxpr = jax.make_jaxpr(f)(jnp.array([1.0, 2.0]), 1.0)

    key = jax.random.PRNGKey(42)
    model_t = hk.without_apply_rng(hk.transform(model))

    x = jnp.array([1.0, 2.0])
    params = model_t.init(key, x)

    jitf = jax.jit(partial(model_t.apply, params))

    import timeit

    # t = timeit.Timer(lambda: jitf(x))
    # print(t.timeit(100000)/100000 * (1000*1000*1000))
    # return

    jaxpr = jax.make_jaxpr(partial(model_t.apply, params))(x)
    # to_cpp(jaxpr, "f")
    # print("=" * 80)
    # print(jaxpr)
    # print("-" * 80)
    print(print_ast(to_cpp(jaxpr, "Foo")))
    # print("=" * 80)


if __name__ == "__main__":
    main()
