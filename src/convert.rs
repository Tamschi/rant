//! # rant::convert
//! Provides conversions between RantValues and native types.

#![allow(unused_mut)]
#![allow(unused_parens)]
#![allow(unused_variables)]

use crate::value::*;
use crate::{
    lang::{Identifier, Parameter, Varity},
    stdlib::RantStdResult,
    RantMapRef, RantString,
};
use crate::{runtime::*, RantList, RantListRef};
use cast::Error as CastError;
use cast::*;
use std::{
    cell::RefCell,
    convert::{TryFrom, TryInto},
    error::Error,
    fmt::Display,
    ops::{Deref, DerefMut},
    rc::Rc,
};

trait ToCastResult<T> {
    fn to_cast_result(self) -> Result<T, CastError>;
}

impl<T> ToCastResult<T> for Result<T, CastError> {
    fn to_cast_result(self) -> Result<T, CastError> {
        self
    }
}

impl ToCastResult<i64> for i64 {
    fn to_cast_result(self) -> Result<i64, CastError> {
        Ok(self)
    }
}

fn rant_cast_error(from: &'static str, to: &'static str, err: CastError) -> ValueError {
    ValueError::InvalidConversion {
        from,
        to,
        message: Some(
            match err {
                CastError::Overflow => "integer overflow",
                CastError::Underflow => "integer underflow",
                CastError::Infinite => "infinity",
                CastError::NaN => "not a number",
            }
            .to_owned(),
        ),
    }
}

macro_rules! impl_infallible_from {
  ($variant:ident <- $first_from:ty $(, $rest_from:ty)*$(,)?) => {
    impl From<$first_from> for RantValue {
      type Output = Self;

      fn cast(value: $first_from) -> Self::Output {
          RantValue::$variant(value.into())
      }
    }

    $(impl_infallible_from!($variant <- $rest_from);)*
  };
}

impl_infallible_from!(String <- char, String, &'_ String, &'_ str, &'_ mut str);
impl_infallible_from!(Float <- f32, f64);
impl_infallible_from!(Integer <- i8, i16, i32, u8, u16, u32); //TODO: NonZero integers?
impl_infallible_from!(Boolean <- bool);

impl<T: Into<RantValue>> From<Option<T>> for RantValue {
    type Output = Self;

    fn cast(value: Option<T>) -> Self::Output {
        value.map_or(RantValue::Empty, Into::into)
    }
}

impl<T: Into<RantValue>> From<Vec<T>> for RantValue {
    type Output = Self;

    fn cast(value: Vec<T>) -> Self::Output {
        RantValue::List(Rc::new(RefCell::new(
            value.into_iter().map(Into::into).collect(),
        )))
    }
}

macro_rules! impl_try_from {
    ($variant:ident($target_type:ty) <- $first_from:ty $(, $rest_from:ty)*$(,)?) => {
        impl TryFrom<$first_from> for RantValue {
          type Error = <$target_type as TryFrom<$first_from>>::Error;

          fn try_from(value: $first_from) -> Result<Self, Self::Error> {
              Ok(RantValue::$variant(value.try_into()?))
          }
        }

        $(impl_try_from!($variant($target_type) <- $rest_from);)*
    };
}

impl_try_from!(Integer(i64) <- i128, isize, u64, u128, usize);

impl<T: TryInto<RantValue>> TryFrom<Option<T>> for RantValue {
    type Error = T::Error;

    fn try_from(value: Option<T>) -> Result<Self, Self::Error> {
        value.map(TryInto::try_into).unwrap_or(Ok(RantValue::Empty))
    }
}

impl<T: TryInto<RantValue>> TryFrom<Vec<T>> for RantValue {
    type Error = T::Error;

    fn try_from(value: Vec<T>) -> Result<Self, Self::Error> {
        Ok(RantValue::List(Rc::new(RefCell::new(
            value
                .into_iter()
                .map(TryInto::try_into)
                .collect::<Result<_, _>>()?,
        ))))
    }
}

impl From<RantValue> for RantEmpty {
    type Output = Self;

    fn cast(_value: RantValue) -> Self::Output {
        Self
    }
}

impl From<RantValue> for String {
    type Output = Self;

    fn cast(value: RantValue) -> Self::Output {
        value.to_string()
    }
}

impl From<RantValue> for Box<str> {
    type Output = Self;

    fn cast(value: RantValue) -> Self::Output {
        let string: String = value.into();
        string.into_boxed_str()
    }
}

pub enum ConversionError<E> {
    IncompatibleType {
        from: &'static str,
        to: &'static str,
        message: Option<String>,
    },
    IncompatibleValue(E),
}

impl<E: Error> Error for ConversionError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            ConversionError::IncompatibleType { .. } => None,
            ConversionError::IncompatibleValue(e) => Some(e),
        }
    }
}

impl<E: Display> Display for ConversionError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConversionError::IncompatibleType { from, to, message } => {
                write!(f, "unable to convert from {} to {}", from, to)?;
                if let Some(message) = message {
                    write!(f, ": {}", message)?
                }
            }
            ConversionError::IncompatibleValue(e) => e.fmt(f)?,
        }
        Ok(())
    }
}

macro_rules! impl_try_from_value {
    ($variant:ident($from_type:ty) -> $($into:ty),+$(,)?) => {
        impl_try_from_value_aliased!($variant($from_type) -> $($into as stringify!($into)),+);
    };
}

macro_rules! impl_try_from_value_aliased {
  ($variant:ident($from_type:ty) -> $first_into:ty as $first_into_to:expr $(, $rest_into:ty as $rest_into_to:expr)*$(,)?) => {
      impl TryFrom<RantValue> for $first_into {
          type Error = ConversionError<<$from_type as TryInto<Self>>::Error>;

          fn try_from(value: RantValue) -> Result<Self, Self::Error> {
              match value {
                  RantValue::$variant(value) => {
                      value.try_into().map_err(ConversionError::IncompatibleValue)
                  }
                  other => ConversionError::IncompatibleType {
                      from: other.type_name(),
                      to: $first_into_to,
                      message: None,
                  },
              }
          }
      }

      $(impl_try_from_value_aliased!($variant($from_type) -> $rest_into as $rest_into_to);)*
  };
}

impl_try_from_value!(Float(f64) -> f32, f64);
impl_try_from_value!(Integer(i64) -> f32, f64, i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);
impl_try_from_value!(Boolean(bool) -> bool);

impl_try_from_value_aliased!(List(RantListRef) -> RantListRef as "list");
impl_try_from_value_aliased!(Map(RantMapRef) -> RantMapRef as "map");
impl_try_from_value_aliased!(Function(RantFunctionRef) -> RantFunctionRef as "function");

trait MaybeOptional: TryFrom<RantValue> {
    const IS_RANT_OPTIONAL: bool;
}

impl<T> TryFrom<RantValue> for Option<T>
where
    RantValue: TryInto<T>,
{
    type Error = <RantValue as TryInto<T>>::Error;

    fn try_from(value: RantValue) -> Result<Self, Self::Error> {
        match value {
            RantValue::Empty => Ok(None),
            other => Some(other.try_into()).transpose(),
        }
    }
}

impl<T: TryFrom<RantValue>> MaybeOptional for Option<T> {
    const IS_RANT_OPTIONAL: bool = true;
}

impl<T, E> TryInto<Vec<T>> for RantValue
where
    RantValue: TryInto<T, Error = ConversionError<E>>,
{
    type Error = ConversionError<E>;

    fn try_into(self) -> Result<Vec<T>, Self::Error> {
        match self {
            RantValue::List(list_ref) => list_ref
                .borrow()
                .iter()
                .cloned()
                .map(TryInto::try_into)
                .collect(),
            other => Err(ConversionError::IncompatibleType {
                from: other.type_name(),
                to: stringify!(Vec<T>),
                message: "only lists can be turned into vectors".to_owned().into(),
            }),
        }
    }
}

#[inline(always)]
fn as_varity<T: FromRant>() -> Varity {
    if T::is_rant_optional() {
        Varity::Optional
    } else {
        Varity::Required
    }
}

#[inline(always)]
fn inc(counter: &mut usize) -> usize {
    let prev = *counter;
    *counter += 1;
    prev
}

/// Converts from argument list to tuple of `impl FromRant` values
pub trait FromRantArgs: Sized {
    fn from_rant_args(args: Vec<RantValue>) -> ValueResult<Self>;
    fn as_rant_params() -> Vec<Parameter>;
}

impl<T: FromRant> FromRantArgs for T {
    fn from_rant_args(args: Vec<RantValue>) -> ValueResult<Self> {
        let mut args = args.into_iter();
        Ok(T::from_rant(args.next().unwrap_or(RantValue::Empty))?)
    }

    fn as_rant_params() -> Vec<Parameter> {
        let varity = if T::is_rant_optional() {
            Varity::Optional
        } else {
            Varity::Required
        };

        let param = Parameter {
            name: Identifier::new(RantString::from("arg0")),
            varity,
        };

        vec![param]
    }
}

/// Semantic wrapper around a Vec<T> for use in optional variadic argument lists.
pub(crate) struct VarArgs<T: FromRant>(Vec<T>);

impl<T: FromRant> VarArgs<T> {
    pub fn new(args: Vec<T>) -> Self {
        Self(args)
    }
}

impl<T: FromRant> Deref for VarArgs<T> {
    type Target = Vec<T>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: FromRant> DerefMut for VarArgs<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Semantic wrapper around a Vec<T> for use in required variadic argument lists.
pub(crate) struct RequiredVarArgs<T: FromRant>(Vec<T>);

impl<T: FromRant> RequiredVarArgs<T> {
    pub fn new(args: Vec<T>) -> Self {
        Self(args)
    }
}

impl<T: FromRant> Deref for RequiredVarArgs<T> {
    type Target = Vec<T>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: FromRant> DerefMut for RequiredVarArgs<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

macro_rules! impl_from_rant_args {
  ($($generic_types:ident),*) => {
    // Non-variadic implementation
    impl<$($generic_types: FromRant,)*> FromRantArgs for ($($generic_types,)*) {
      fn from_rant_args(args: Vec<RantValue>) -> ValueResult<Self> {
        let mut args = args.into_iter();
        Ok(($($generic_types::from_rant(args.next().unwrap_or(RantValue::Empty))?,)*))
      }

      fn as_rant_params() -> Vec<Parameter> {
        let mut i: usize = 0;
        vec![$(Parameter {
          name: Identifier::new(RantString::from(format!("arg{}", inc(&mut i)))),
          varity: as_varity::<$generic_types>(),
        },)*]
      }
    }

    // Variadic* implementation
    impl<$($generic_types: FromRant,)* VarArgItem: FromRant> FromRantArgs for ($($generic_types,)* VarArgs<VarArgItem>) {
      fn from_rant_args(mut args: Vec<RantValue>) -> ValueResult<Self> {
        let mut args = args.drain(..);
        Ok(
          ($($generic_types::from_rant(args.next().unwrap_or(RantValue::Empty))?,)*
          VarArgs::new(args
            .map(VarArgItem::from_rant)
            .collect::<ValueResult<Vec<VarArgItem>>>()?
          )
        ))
      }

      fn as_rant_params() -> Vec<Parameter> {
        let mut i: usize = 0;
        vec![$(Parameter {
          name: Identifier::new(RantString::from(format!("arg{}", inc(&mut i)))),
          varity: as_varity::<$generic_types>(),
        },)*
        Parameter {
          name: Identifier::new(RantString::from(format!("arg{}", inc(&mut i)))),
          varity: Varity::VariadicStar,
        }]
      }
    }

    // Variadic+ implementation
    impl<$($generic_types: FromRant,)* VarArgItem: FromRant> FromRantArgs for ($($generic_types,)* RequiredVarArgs<VarArgItem>) {
      fn from_rant_args(mut args: Vec<RantValue>) -> ValueResult<Self> {
        let mut args = args.drain(..);
        Ok(
          ($($generic_types::from_rant(args.next().unwrap_or(RantValue::Empty))?,)*
          RequiredVarArgs::new(args
            .map(VarArgItem::from_rant)
            .collect::<ValueResult<Vec<VarArgItem>>>()?
          )
        ))
      }

      fn as_rant_params() -> Vec<Parameter> {
        let mut i: usize = 0;
        vec![$(Parameter {
          name: Identifier::new(RantString::from(format!("arg{}", inc(&mut i)))),
          varity: as_varity::<$generic_types>(),
        },)*
        Parameter {
          name: Identifier::new(RantString::from(format!("arg{}", inc(&mut i)))),
          varity: Varity::VariadicPlus,
        }]
      }
    }
  }
}

impl_from_rant_args!();
impl_from_rant_args!(A);
impl_from_rant_args!(A, B);
impl_from_rant_args!(A, B, C);
impl_from_rant_args!(A, B, C, D);
impl_from_rant_args!(A, B, C, D, E);
impl_from_rant_args!(A, B, C, D, E, F);
impl_from_rant_args!(A, B, C, D, E, F, G);
impl_from_rant_args!(A, B, C, D, E, F, G, H);
impl_from_rant_args!(A, B, C, D, E, F, G, H, I);
impl_from_rant_args!(A, B, C, D, E, F, G, H, I, J);
impl_from_rant_args!(A, B, C, D, E, F, G, H, I, J, K);
//impl_from_rant_args!(A, B, C, D, E, F, G, H, I, J, K, L);

/// Trait for converting something to a Rant function.
pub trait AsRantFunction<Params: FromRantArgs> {
    /// Performs the conversion.
    fn as_rant_func(&'static self) -> RantFunction;
}

impl<Params: FromRantArgs, Function: Fn(&mut VM, Params) -> RantStdResult> AsRantFunction<Params>
    for Function
{
    fn as_rant_func(&'static self) -> RantFunction {
        let body = RantFunctionInterface::Foreign(Rc::new(move |vm, args| {
            self(vm, Params::from_rant_args(args).into_runtime_result()?)
        }));

        let params = Rc::new(Params::as_rant_params());

        RantFunction {
            body,
            captured_vars: vec![],
            min_arg_count: params.iter().take_while(|p| p.is_required()).count(),
            vararg_start_index: params
                .iter()
                .enumerate()
                .find_map(|(i, p)| {
                    if p.varity.is_variadic() {
                        Some(i)
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| params.len()),
            params,
        }
    }
}
