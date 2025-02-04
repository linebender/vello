use rune::ast;
use rune::ast::{
    Ident,
    ByteIndex,
    Path,
    Pat,
    PatPath,
    Span,
    Stmt::{
        Local,
    },
};
use rune::SourceId;
use rune::parse::Parser;
use rune::alloc::Box;
use rune::alloc::clone::TryClone;

use std::collections::HashMap;

pub struct CodeGen {
    pub content: String,
    pub vune_path: Option<String>,
    pub ext_content: HashMap<String, String>,
    pub full_names: HashMap<String, String>,
    pub current_struct_name: String,
}

pub type RuneVec<T> = rune::alloc::Vec<T>;

impl CodeGen {
    pub fn new(content: &str, vune_path: &str) -> Self {
        let mut ext_content = HashMap::new();

        ext_content.insert("core::transform".to_string(), include_str!("../core/transform.vune").to_string());
        ext_content.insert("core::points".to_string(), include_str!("../core/points.vune").to_string());

        Self {
            content: content.to_string(),
            vune_path: match vune_path != "" {
                true => Some(vune_path.to_string()),
                false => None,
            },
            ext_content,
            full_names: HashMap::new(),
            current_struct_name: String::new(),
        }
    }

    pub fn codegen_stmt(&mut self, stmt: ast::Stmt) -> String {
        match stmt {
            ast::Stmt::Local(l) => self.codegen_local(l),
            ast::Stmt::Expr(e) => self.codegen_expr(e),
            ast::Stmt::Semi(s) => self.codegen_stmt_semi(s),
            _ => { dbg!(stmt); todo!() },
        }
    }

    pub fn codegen_stmt_semi(&mut self, semi: ast::StmtSemi) -> String {
        self.codegen_expr(semi.expr) + ";"
    }

    pub fn codegen_expr(&mut self, expr: ast::Expr) -> String {
        match expr {
            ast::Expr::Lit(ast::ExprLit { lit, .. }) => self.codegen_lit(lit),
            ast::Expr::Binary(b) => self.codegen_binary(b),
            ast::Expr::Path(p) => self.codegen_path(p),
            ast::Expr::Return(r) => self.codegen_return(r),
            ast::Expr::Assign(a) => self.codegen_assign(a),
            ast::Expr::FieldAccess(f) => self.codegen_field_access(f),
            ast::Expr::Call(c) => self.codegen_call(c),
            ast::Expr::Group(g) => self.codegen_group(g),
            ast::Expr::Unary(u) => self.codegen_unary(u),
            _ => { dbg!(expr); todo!() },
        }
    }

    pub fn codegen_unary(&mut self, unary: ast::ExprUnary) -> String {
        self.codegen_unop(unary.op) + &self.codegen_expr(Box::into_inner(unary.expr))
    }

    pub fn codegen_unop(&mut self, unop: ast::UnOp) -> String {
        match unop {
            ast::UnOp::Neg(_) => "-".to_string(),
            _ => { dbg!(unop); todo!() },
        }
    }

    pub fn codegen_group(&mut self, group: ast::ExprGroup) -> String {
        "(".to_owned() + &self.codegen_expr(Box::into_inner(group.expr)) + ")"
    }

    pub fn codegen_call(&mut self, call: ast::ExprCall) -> String {
        self.codegen_expr(Box::into_inner(call.expr)).replace("::", "_") + 
        "(" +
        &self.codegen_call_args(call.args) +
        ")"
    }

    pub fn codegen_field_access(&mut self, field: ast::ExprFieldAccess) -> String {
        let expr = self.codegen_expr(Box::into_inner(field.expr));
        let field = self.codegen_expr_field(field.expr_field);

        expr.to_string() + "." + &field
    }

    pub fn codegen_expr_field(&mut self, expr_field: ast::ExprField) -> String {
        match expr_field {
            ast::ExprField::Path(p) => self.codegen_path(p),
            ast::ExprField::LitNumber(l) => self.codegen_lit_number(l),
            _ => todo!(),
        }
    }

    pub fn codegen_lit_number(&mut self, lit: ast::LitNumber) -> String {
        // TODO: add NumberSource (source)
        self.span_to_str(lit.span)
    }

    pub fn codegen_assign(&mut self, assign: ast::ExprAssign) -> String {
        let lhs = self.codegen_expr(Box::into_inner(assign.lhs));
        let rhs = self.codegen_expr(Box::into_inner(assign.rhs));

        lhs.to_string() + " = " + &rhs
    }

    pub fn codegen_return(&mut self, ret: ast::ExprReturn) -> String {
        "return ".to_owned() + &self.codegen_expr(Box::into_inner(ret.expr.unwrap()))
    }

    pub fn codegen_binary(&mut self, bin: ast::ExprBinary) -> String {
        let lhs = self.codegen_expr(Box::into_inner(bin.lhs));
        let rhs = self.codegen_expr(Box::into_inner(bin.rhs));
        let op = self.codegen_binary_operator(bin.op);

        lhs + " " + &op + " " + &rhs
    }

    pub fn codegen_binary_operator(&mut self, op: ast::BinOp) -> String {
        match op {
            ast::BinOp::Add(_) => "+".to_string(),
            ast::BinOp::Sub(_) => "-".to_string(),
            ast::BinOp::Mul(_) => "*".to_string(),
            ast::BinOp::Div(_) => "/".to_string(),
            ast::BinOp::AddAssign(_) => "+=".to_string(),
            ast::BinOp::SubAssign(_) => "-=".to_string(),
            ast::BinOp::MulAssign(_) => "*=".to_string(),
            ast::BinOp::DivAssign(_) => "/=".to_string(),
            _ => todo!(),
        }
    }

    pub fn codegen_lit(&mut self, lit: ast::Lit) -> String {
        match lit {
            ast::Lit::Number(l) => self.codegen_lit_number(l),
            _ => todo!(),
        }
    }
    
    pub fn codegen_local(&mut self, local: Box<ast::Local>) -> String {
        let local_name = self.codegen_pat(local.pat.try_clone().unwrap());
        let local_expr = self.codegen_expr(local.expr.try_clone().unwrap());

        let final_local = if local.mut_token.is_some() {
            "var "
        }
        else {
            "let "
        };

        final_local.to_owned() + &local_name + " = " + &local_expr + ";"
    }

    pub fn codegen_pat(&mut self, pat: Pat) -> String {
        match pat {
            ast::Pat::Path(p) => self.get_first_name_from_path(p.path),
            ast::Pat::Binding(b) => self.codegen_pat_binding(b),
            _ => todo!(),
        }
    }

    pub fn codegen_pat_binding(&mut self, pb: ast::PatBinding) -> String {
        let key = self.codegen_object_key(pb.key);
        let pat = self.codegen_pat(Box::into_inner(pb.pat));

        key + ": " + &pat
    }

    pub fn codegen_object_key(&mut self, ok: ast::ObjectKey) -> String {
        match ok {
            ast::ObjectKey::Path(p) => self.codegen_path(p),
            _ => todo!(),
        }
    }

    pub fn get_first_name_from_pat(&mut self, pat: Pat) -> String {
        match pat {
            rune::ast::Pat::Path(p) => self.get_first_name_from_pat_path(p),
            _ => todo!(),
        }
    }

    pub fn get_first_name_from_pat_path(&mut self, pat: PatPath) -> String {
        self.get_first_name_from_path(pat.path)
    }

    pub fn get_first_name_from_path(&mut self, p: Path) -> String {
        match p.first {
            rune::ast::PathSegment::Ident(id) => self.ident_to_str(id, true),
            _ => todo!(),
        }
    }

    pub fn ident_to_str(&mut self, id: ast::Ident, full_name: bool) -> String {
        let mut result = self.span_to_str(id.span);
        let mut final_result = String::new();

        match full_name {
            true => { 
                match self.full_names.get(&result) {
                    Some(r) => final_result = r.replace("::", "_"),
                    None => final_result = result,
                }

                final_result
            },
            false => result,
        }
    }

    pub fn codegen_function(&mut self, function: ast::ItemFn, impl_name: &Option<String>) -> String {
        let mut function_name = match impl_name {
            Some(im) => im.replace("::", "_") + "_" + &self.ident_to_str(function.name, true),
            None => self.ident_to_str(function.name, true),
        };

        let block = self.codegen_block(function.body, 1);
        let args = self.codegen_args(function.args);

        let mut arrow_with_type = if function.output.is_some() {
            "-> ".to_owned() + &self.codegen_type(function.output.unwrap().1)
        }
        else {
            "".to_string()
        };

        "fn ".to_owned() + &function_name + "(" + &args + ") " + &arrow_with_type + " {\n"
            + &block +
        "}\n"
    }

    pub fn codegen_args(&mut self, args: ast::Parenthesized<ast::FnArg, ast::Comma>) -> String {
        let mut result = String::new();
        let length: isize = args.parenthesized.len() as isize;

        let mut count = 0;
        for i in args.parenthesized {
            match i.0 {
                ast::FnArg::Pat(p) => result += &self.codegen_pat(p),
                _ => todo!(),
            }

            if count < length - 1 {
                result += ", ";
                count += 1;
            }
        }

        result
    }

    pub fn codegen_call_args(&mut self, args: ast::Parenthesized<ast::Expr, ast::Comma>) -> String {
        let mut result = String::new();
        let length: isize = args.parenthesized.len() as isize;

        let mut count = 0;
        for i in args.parenthesized {
            result += &self.codegen_expr(i.0);

            if count < length - 1 {
                result += ", ";
                count += 1;
            }
        }

        result
    }

    pub fn get_tabs(&mut self, tabs: usize) -> String {
        let mut result = String::new();

        for _i in 0..tabs {
            result += "    ";
        }

        result
    }

    pub fn codegen_block(&mut self, block: ast::Block, tabs: usize) -> String {
        let mut result = String::new();

        for stmt in block.statements {
            result += &(self.get_tabs(tabs) + &self.codegen_stmt(stmt) + "\n");
        }

        result
    }

    pub fn span_to_str(&mut self, span: Span) -> String {
        self.content[span.start.into_usize()..span.end.into_usize()].to_string()
    }

    pub fn add_file(&mut self, filename: &str, input: &str) -> String {
        self.add_str(&std::fs::read_to_string(filename).unwrap(), input)
    }

    pub fn add_str(&mut self, add: &str, to: &str) -> String {
        let mut res = add.to_string();

        res += "\n";
        res += to;

        res
    }

    pub fn codegen_use(&mut self, use_item: ast::ItemUse) -> String {
        let mut item_path = self.codegen_item_use_path(use_item.path);

        let cont = self.ext_content.get(&item_path);
        let sh = VuneShader::new(&cont.unwrap(), &item_path);

        self.full_names.extend(sh.full_names.clone());

        sh.content()
    }

    pub fn codegen_item_use_path(&mut self, use_path: ast::ItemUsePath) -> String {
        let mut result = self.codegen_item_use_segment(use_path.first);

        for i in use_path.segments {
            result += &("::".to_owned() + &self.codegen_item_use_segment(i.1));
        }

        result
    }

    pub fn codegen_item_use_segment(&mut self, ius: ast::ItemUseSegment) -> String {
        match ius {
            ast::ItemUseSegment::PathSegment(p) => self.codegen_path_segment(p),
            _ => { dbg!(ius); todo!() },
        }
    }

    pub fn codegen_path_segment(&mut self, path_seg: ast::PathSegment) -> String {
        match path_seg {
            ast::PathSegment::Ident(i) => self.ident_to_str(i, true),
            ast::PathSegment::SelfType(s) => self.codegen_self(),
            _ => { dbg!(path_seg); todo!() },
        }
    }

    pub fn codegen_self(&mut self) -> String {
        self.current_struct_name.clone()
    }

    pub fn codegen_file(&mut self, file: ast::File) -> String {
        let mut result = String::new();

        for i in file.items {
            result += &(self.codegen_item(i.0) + "\n");
        }

        result
    }

    pub fn codegen_item(&mut self, item: ast::Item) -> String {
        match item {
            ast::Item::Fn(f) => self.codegen_function(f, &None),
            ast::Item::Use(u) => self.codegen_use(u),
            ast::Item::Impl(i) => self.codegen_impl(i),
            ast::Item::Struct(s) => self.codegen_struct(s),
            ast::Item::MacroCall(m) => self.codegen_macro(m),
            _ => { dbg!(item); todo!() }
        }
    }

    pub fn codegen_macro(&mut self, macro_call: ast::MacroCall) -> String {
        let name: &str = &self.codegen_path(macro_call.path.try_clone().unwrap());

        match name {
            "uniform" => self.codegen_uniform(macro_call),
            _ => { dbg!(macro_call); todo!() },
        }
    }

    pub fn codegen_uniform(&mut self, macro_call: ast::MacroCall) -> String {
        let input_vec: &RuneVec<ast::Token> = macro_call.input.vec();
        let mut numb_vec: RuneVec<ast::Token> = RuneVec::new();
        numb_vec.try_push(input_vec[0]);

        let mut value_vec: RuneVec<ast::Token> = RuneVec::new();
        value_vec.try_extend_from_slice(&input_vec[2..input_vec.len()]);

        let numb_stream = rune::macros::TokenStream::from(numb_vec);
        let mut numb_parser = Parser::from_token_stream(&numb_stream, macro_call.open.span);
        let numb_ast = numb_parser.parse::<ast::LitNumber>().unwrap();
        let numb_codegen = self.codegen_lit_number(numb_ast);

        let value_stream = rune::macros::TokenStream::from(value_vec);
        let mut value_parser = Parser::from_token_stream(&value_stream, macro_call.open.span);
        let value_ast = value_parser.parse::<ast::Field>().unwrap();
        let value_codegen = self.codegen_field(value_ast);

        "@group(0) @binding(".to_owned() + &numb_codegen + ")\n" +
        "var<uniform> " + &value_codegen + ";\n"
    }

    pub fn codegen_struct(&mut self, struct_item: ast::ItemStruct) -> String {
        let name = self.ident_to_str(struct_item.ident, false);

        let mut full_name = match &self.vune_path {
            Some(p) => p.to_owned() + "::" + &self.ident_to_str(struct_item.ident, false),
            None => self.ident_to_str(struct_item.ident, false),
        };

        self.full_names.insert(name, full_name.clone());

        "struct ".to_owned() + &full_name.replace("::", "_") + " {\n" + &self.codegen_fields_body(struct_item.body) + "}\n"
    }

    pub fn codegen_fields_body(&mut self, fields: ast::Fields) -> String {
        let mut result = String::new();

        match fields {
            ast::Fields::Named(b) => {
                for i in b.braced {
                    result += &(self.get_tabs(1) + &self.codegen_field(i.0));

                    match i.1 {
                        Some(_) => result += ",\n",
                        None => result += "\n",
                    }
                }
            },
            _ => { dbg!(fields); todo!() },
        }

        result
    }

    pub fn codegen_field(&mut self, field: ast::Field) -> String {
        let name = self.ident_to_str(field.name, false);
        let field_type = match field.ty {
            Some(ft) => ": ".to_owned() + &self.codegen_type(ft.1),
            None => "".to_string(),
        };

        name + &field_type
    }

    pub fn codegen_type(&mut self, ast_type: ast::Type) -> String {
        match ast_type {
            ast::Type::Path(p) => self.codegen_path(p),
            _ => { dbg!(ast_type); todo!() },
        }
    }

    pub fn codegen_impl(&mut self, impl_ast: ast::ItemImpl) -> String {

        let mut result = String::new();

        let impl_path = self.codegen_path(impl_ast.path);

        self.current_struct_name = impl_path.clone();

        let option_impl = Some(impl_path);
        for i in impl_ast.functions {
            result += &self.codegen_function(i, &option_impl);
        }

        self.current_struct_name.clear();

        result
    }

    pub fn codegen_path(&mut self, p: ast::Path) -> String {
        let mut result = self.codegen_path_segment(p.first);

        for i in p.rest {
            result += &("::".to_owned() + &self.codegen_path_segment(i.1));
        }

        result
    }
}

pub struct VuneShader {
    content: String,
    vune_path: Option<String>,
    pub full_names: HashMap<String, String>,
}

fn parse_and_codegen(input: &str, codegen: &mut CodeGen) -> (String, HashMap<String, String>) {
    let mut parser = Parser::new(input, SourceId::empty(), false);

    let mut ast = parser.parse_all::<ast::File>().unwrap();

    (codegen.codegen_file(ast), codegen.full_names.clone())
}

impl VuneShader {
    pub fn new(input: &str, vune_path: &str) -> Self {
        let mut codegen = CodeGen::new(&input, vune_path);
        let result = parse_and_codegen(input, &mut codegen);

        Self {
            content: result.0,
            vune_path: match vune_path != "" {
                true => Some(vune_path.to_string()),
                false => None,
            },
            full_names: result.1,
        }
    }

    pub fn new_main(input: &str) -> Self {
        let file = include_str!("../base_shaders/flatten.wgsl");

        let mut vune = VuneShader::new(input, "");

        let cont = vune.content_mut_ref();
        *cont = file.to_string() + "\n" + cont;

        vune
    }

    pub fn new_from_file(file: &str, vune_path: &str) -> Self {
        VuneShader::new(&std::fs::read_to_string(file).unwrap(), vune_path)
    }

    pub fn new_main_from_file(file: &str) -> Self {
        VuneShader::new_main(&std::fs::read_to_string(file).unwrap())
    }

    pub fn content(&self) -> String {
        self.content.clone()
    }

    pub fn content_ref(&self) -> &String {
        &self.content
    }

    pub fn content_mut_ref(&mut self) -> &mut String {
        &mut self.content
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        println!("{}", VuneShader::new_main_from_file("test.vune").content());
    }
}
