from __future__ import annotations

import dataclasses
from typing import List, Optional, Union


@dataclasses.dataclass
class Node:
    pass


@dataclasses.dataclass
class Parameter(Node):
    typename: str
    name: str


@dataclasses.dataclass
class LiteralExpr(Node):
    literal: str


@dataclasses.dataclass
class VarExpr(Node):
    name: str


@dataclasses.dataclass
class CallExpr(Node):
    name: str
    args: List[Expr]


@dataclasses.dataclass
class BinaryOpExpr(Node):
    op: str
    lhs: Expr
    rhs: Expr


@dataclasses.dataclass
class AssignmentExpr(Node):
    typename: str
    name: str
    expr: Expr


Expr = AssignmentExpr | BinaryOpExpr | CallExpr | LiteralExpr | VarExpr


@dataclasses.dataclass
class ReturnStmt(Node):
    expr: Expr


@dataclasses.dataclass
class ExprStmt(Node):
    expr: Expr


Stmt = ReturnStmt | ExprStmt


@dataclasses.dataclass
class Function(Node):
    name: str
    typename: str
    parameters: List[Parameter]
    body: List[Stmt]

    modifiers: List[str] = dataclasses.field(default_factory=lambda: [])


@dataclasses.dataclass
class ClassDecl(Node):
    template: str

    name: str

    types: List[Stmt]

    constructor: Optional[Function]

    members: List[Stmt]
    methods: List[Function]


@dataclasses.dataclass
class Namespace(Node):
    name: str
    decls: List[Decl]


Decl = Function | ClassDecl | Namespace


@dataclasses.dataclass
class Directive(Node):
    kind: str
    value: str


@dataclasses.dataclass
class Module(Node):
    name: str

    directives: List[Directive]
    decls: List[Decl]
