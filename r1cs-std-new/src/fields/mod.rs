use algebra::{fields::BitIterator, Field};
use core::fmt::Debug;
use r1cs_core::{ConstraintSystem, SynthesisError};

use crate::{prelude::*, Assignment};

pub mod fp;
// pub mod fp12;
// pub mod fp2;
// pub mod fp3;
// pub mod fp4;
// pub mod fp6_2over3;
// pub mod fp6_3over2;

pub trait FieldGadget<F: Field, ConstraintF: Field>:
    Sized
    + Clone
    // + EqGadget<ConstraintF>
    // + NEqGadget<ConstraintF>
    // + ConditionalEqGadget<ConstraintF>
    // + ToBitsGadget<ConstraintF>
    // + AllocGadget<F, ConstraintF>
    // + ToBytesGadget<ConstraintF>
    // + CondSelectGadget<ConstraintF>
    // + TwoBitLookupGadget<ConstraintF, TableConstant = F>
    // + ThreeBitCondNegLookupGadget<ConstraintF, TableConstant = F>
    + std::ops::Neg<Output = Self>
    + for<'a> std::ops::Add<&'a Self, Output = Self>
    + for<'a> std::ops::Sub<&'a Self, Output = Self>
    + for<'a> std::ops::Mul<&'a Self, Output = Self>
    + for<'a> std::ops::AddAssign<&'a Self>
    + for<'a> std::ops::SubAssign<&'a Self>
    + for<'a> std::ops::MulAssign<&'a Self>
    + for<'a> std::ops::Add<F, Output = Self>
    + for<'a> std::ops::Sub<F, Output = Self>
    + for<'a> std::ops::Mul<F, Output = Self>
    + for<'a> std::ops::AddAssign<F>
    + for<'a> std::ops::SubAssign<F>
    + for<'a> std::ops::MulAssign<F>
    + Debug
{
    fn zero<CS: ConstraintSystem<ConstraintF>>(_: CS) -> Result<Self, SynthesisError>;

    fn one<CS: ConstraintSystem<ConstraintF>>(_: CS) -> Result<Self, SynthesisError>;

    // fn conditionally_add_constant(
    //     &self,
    //     _: &Boolean,
    //     _: F,
    // ) -> Result<Self, SynthesisError>;

    /// Output `2 * self`.
    fn double(&self) -> Result<Self, SynthesisError> {
        let mut result = self.clone();
        result.double_in_place();
        Ok(result)
    }

    /// Replace `self` with `2 * self`.
    fn double_in_place(&mut self) -> Result<&mut Self, SynthesisError> {
        *self += &*self;
        Ok(self)
    }

    /// Replace `self` with `-self`.
    #[inline]
    fn negate_in_place(&mut self) -> Result<&mut Self, SynthesisError> {
        *self = -self.clone();
        Ok(self)
    }

    /// Output `self^2`.
    fn square(&self, cs: CS) -> Result<Self, SynthesisError> {
        let mut result = self.clone();
        result.square_in_place();
        Ok(result)
    }

    /// Replace `self` with `self^2`.
    fn square_in_place(&mut self) -> Result<&mut Self, SynthesisError> {
        *self *= &*self;
        Ok(self)
    }

    /// Enforce that `self * other = result`.
    fn mul_equals(&self, other: &Self, result: &Self) -> Result<(), SynthesisError> {
        let actual_result = self * other;
        result.enforce_equal(&actual_result)
    }

    fn inverse(&self) -> Result<Self, SynthesisError> {
        let one = Self::one(&mut cs.ns(|| "one"))?;
        let inverse = Self::alloc(&mut cs.ns(|| "alloc inverse"), || {
            self.get_value().and_then(|val| val.inverse()).get()
        })?;
        self.mul_equals(&inverse, &one)?;
        Ok(inverse)
    }

    fn frobenius_map(&self, power: usize) -> Result<Self, SynthesisError>;

    fn frobenius_map_in_place(&mut self, power: usize) -> Result<&mut Self, SynthesisError> {
        *self = self.frobenius_map(power)?;
        Ok(self)
    }

    /// Accepts as input a list of bits which, when interpreted in big-endian
    /// form, are a scalar.
    #[inline]
    fn pow(&self, bits: &[Boolean]) -> Result<Self, SynthesisError> {
        let mut res = Self::one(cs.ns(|| "Alloc result"))?;
        for (i, bit) in bits.iter().enumerate() {
            res = res.square(cs.ns(|| format!("Double {}", i)))?;
            let tmp = res.mul(cs.ns(|| format!("Add {}-th base power", i)), self)?;
            res = Self::conditionally_select(
                cs.ns(|| format!("Conditional Select {}", i)),
                bit,
                &tmp,
                &res,
            )?;
        }
        Ok(res)
    }

    fn pow_by_constant<S: AsRef<[u64]>>(&self, exp: S) -> Result<Self, SynthesisError> {
        let mut res = Self::one(self.cs.clone())?;
        let mut found_one = false;

        for (i, bit) in BitIterator::new(exp).enumerate() {
            if found_one {
                res.square_in_place()?;
            }

            if bit {
                found_one = true;
                res *= self;
            }
        }

        Ok(res)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use rand::{self, SeedableRng};
    use rand_xorshift::XorShiftRng;

    use crate::{prelude::*, test_constraint_system::TestConstraintSystem, Vec};
    use algebra::{test_rng, BitIterator, Field, UniformRand};
    use r1cs_core::ConstraintSystem;

    #[allow(dead_code)]
    pub(crate) fn field_test<FE: Field, ConstraintF: Field, F: FieldGadget<FE, ConstraintF>>() {
        let mut cs = TestConstraintSystem::<ConstraintF>::new();

        let mut rng = test_rng();
        let a_native = FE::rand(&mut rng);
        let b_native = FE::rand(&mut rng);
        let a = F::alloc(&mut cs.ns(|| "generate_a"), || Ok(a_native)).unwrap();
        let b = F::alloc(&mut cs.ns(|| "generate_b"), || Ok(b_native)).unwrap();

        let zero = F::zero(cs.ns(|| "zero")).unwrap();
        let zero_native = zero.get_value().unwrap();
        zero.enforce_equal(&mut cs.ns(|| "zero_equals?"), &zero)
            .unwrap();
        assert_eq!(zero, zero);

        let one = F::one(cs.ns(|| "one")).unwrap();
        let one_native = one.get_value().unwrap();
        assert_eq!(one, one);
        one.enforce_equal(&mut cs.ns(|| "one_equals?"), &one)
            .unwrap();
        assert_ne!(one, zero);

        let one_dup = zero.add(cs.ns(|| "zero_plus_one"), &one).unwrap();
        one_dup
            .enforce_equal(&mut cs.ns(|| "one_plus_zero_equals"), &one)
            .unwrap();
        assert_eq!(one_dup, one);

        let two = one.add(cs.ns(|| "one_plus_one"), &one).unwrap();
        two.enforce_equal(&mut cs.ns(|| "two_equals?"), &two)
            .unwrap();
        assert_eq!(two, two);
        assert_ne!(zero, two);
        assert_ne!(one, two);

        // a == a
        assert_eq!(a, a);

        // a + 0 = a
        let a_plus_zero = a.add(cs.ns(|| "a_plus_zero"), &zero).unwrap();
        assert_eq!(a_plus_zero, a);
        assert_eq!(a_plus_zero.get_value().unwrap(), a_native);
        a_plus_zero
            .enforce_equal(&mut cs.ns(|| "a_plus_zero_equals?"), &a)
            .unwrap();

        // a - 0 = a
        let a_minus_zero = a.sub(cs.ns(|| "a_minus_zero"), &zero).unwrap();
        assert_eq!(a_minus_zero, a);
        assert_eq!(a_minus_zero.get_value().unwrap(), a_native);
        a_minus_zero
            .enforce_equal(&mut cs.ns(|| "a_minus_zero_equals?"), &a)
            .unwrap();

        // a - a = 0
        let a_minus_a = a.sub(cs.ns(|| "a_minus_a"), &a).unwrap();
        assert_eq!(a_minus_a, zero);
        assert_eq!(a_minus_a.get_value().unwrap(), zero_native);
        a_minus_a
            .enforce_equal(&mut cs.ns(|| "a_minus_a_equals?"), &zero)
            .unwrap();

        // a + b = b + a
        let a_b = a.add(cs.ns(|| "a_plus_b"), &b).unwrap();
        let b_a = b.add(cs.ns(|| "b_plus_a"), &a).unwrap();
        assert_eq!(a_b, b_a);
        assert_eq!(a_b.get_value().unwrap(), a_native + &b_native);
        a_b.enforce_equal(&mut cs.ns(|| "a+b == b+a"), &b_a)
            .unwrap();

        // (a + b) + a = a + (b + a)
        let ab_a = a_b.add(cs.ns(|| "a_b_plus_a"), &a).unwrap();
        let a_ba = a.add(cs.ns(|| "a_plus_b_a"), &b_a).unwrap();
        assert_eq!(ab_a, a_ba);
        assert_eq!(ab_a.get_value().unwrap(), a_native + &b_native + &a_native);
        ab_a.enforce_equal(&mut cs.ns(|| "a+b + a == a+ b+a"), &a_ba)
            .unwrap();

        let b_times_a_plus_b = a_b.mul(cs.ns(|| "b * (a + b)"), &b).unwrap();
        let b_times_b_plus_a = b_a.mul(cs.ns(|| "b * (b + a)"), &b).unwrap();
        assert_eq!(b_times_b_plus_a, b_times_a_plus_b);
        assert_eq!(
            b_times_a_plus_b.get_value().unwrap(),
            b_native * &(b_native + &a_native)
        );
        assert_eq!(
            b_times_a_plus_b.get_value().unwrap(),
            (b_native + &a_native) * &b_native
        );
        assert_eq!(
            b_times_a_plus_b.get_value().unwrap(),
            (a_native + &b_native) * &b_native
        );
        b_times_b_plus_a
            .enforce_equal(&mut cs.ns(|| "b*(a+b) == b * (b+a)"), &b_times_a_plus_b)
            .unwrap();

        // a * 0 = 0
        assert_eq!(a.mul(cs.ns(|| "a_times_zero"), &zero).unwrap(), zero);

        // a * 1 = a
        assert_eq!(a.mul(cs.ns(|| "a_times_one"), &one).unwrap(), a);
        assert_eq!(
            a.mul(cs.ns(|| "a_times_one2"), &one)
                .unwrap()
                .get_value()
                .unwrap(),
            a_native * &one_native
        );

        // a * b = b * a
        let ab = a.mul(cs.ns(|| "a_times_b"), &b).unwrap();
        let ba = b.mul(cs.ns(|| "b_times_a"), &a).unwrap();
        assert_eq!(ab, ba);
        assert_eq!(ab.get_value().unwrap(), a_native * &b_native);

        // (a * b) * a = a * (b * a)
        let ab_a = ab.mul(cs.ns(|| "ab_times_a"), &a).unwrap();
        let a_ba = a.mul(cs.ns(|| "a_times_ba"), &ba).unwrap();
        assert_eq!(ab_a, a_ba);
        assert_eq!(ab_a.get_value().unwrap(), a_native * &b_native * &a_native);

        let aa = a.mul(cs.ns(|| "a * a"), &a).unwrap();
        let a_squared = a.square(cs.ns(|| "a^2")).unwrap();
        a_squared
            .enforce_equal(&mut cs.ns(|| "a^2 == a*a"), &aa)
            .unwrap();
        assert_eq!(aa, a_squared);
        assert_eq!(aa.get_value().unwrap(), a_native.square());

        let aa = a
            .mul_by_constant(cs.ns(|| "a * a via mul_by_const"), &a.get_value().unwrap())
            .unwrap();
        a_squared
            .enforce_equal(&mut cs.ns(|| "a^2 == a*a via mul_by_const"), &aa)
            .unwrap();
        assert_eq!(aa, a_squared);
        assert_eq!(aa.get_value().unwrap(), a_native.square());

        let a_b2 = a
            .add_constant(cs.ns(|| "a + b via add_const"), &b.get_value().unwrap())
            .unwrap();
        a_b.enforce_equal(&mut cs.ns(|| "a + b == a + b via add_const"), &a_b2)
            .unwrap();
        assert_eq!(a_b, a_b2);

        let a_inv = a.inverse(cs.ns(|| "a_inv")).unwrap();
        a_inv
            .mul_equals(cs.ns(|| "check_equals"), &a, &one)
            .unwrap();
        assert_eq!(
            a_inv.get_value().unwrap(),
            a.get_value().unwrap().inverse().unwrap()
        );
        assert_eq!(a_inv.get_value().unwrap(), a_native.inverse().unwrap());
        // a * a * a = a^3
        let bits = BitIterator::new([0x3])
            .map(Boolean::constant)
            .collect::<Vec<_>>();
        assert_eq!(
            a_native * &(a_native * &a_native),
            a.pow(cs.ns(|| "test_pow"), &bits)
                .unwrap()
                .get_value()
                .unwrap()
        );

        // a * a * a = a^3
        assert_eq!(
            a_native * &(a_native * &a_native),
            a.pow_by_constant(cs.ns(|| "test_constant_pow"), &[3])
                .unwrap()
                .get_value()
                .unwrap()
        );

        // a * a * a = a^3
        let mut constants = [FE::zero(); 4];
        for c in &mut constants {
            *c = UniformRand::rand(&mut test_rng());
            println!("Current c[i]: {:?}", c);
        }
        let bits = [Boolean::constant(false), Boolean::constant(true)];
        let lookup_result =
            F::two_bit_lookup(cs.ns(|| "Lookup"), &bits, constants.as_ref()).unwrap();
        assert_eq!(lookup_result.get_value().unwrap(), constants[2]);

        let negone: FE = UniformRand::rand(&mut test_rng());

        let n = F::alloc(&mut cs.ns(|| "alloc new var"), || Ok(negone)).unwrap();
        let _ = n.to_bytes(&mut cs.ns(|| "ToBytes")).unwrap();
        let _ = n
            .to_non_unique_bytes(&mut cs.ns(|| "ToBytes Strict"))
            .unwrap();

        let ab_false = a
            .conditionally_add_constant(
                cs.ns(|| "Add bool with coeff false"),
                &Boolean::constant(false),
                b_native,
            )
            .unwrap();
        assert_eq!(ab_false.get_value().unwrap(), a_native);
        let ab_true = a
            .conditionally_add_constant(
                cs.ns(|| "Add bool with coeff true"),
                &Boolean::constant(true),
                b_native,
            )
            .unwrap();
        assert_eq!(ab_true.get_value().unwrap(), a_native + &b_native);

        if !cs.is_satisfied() {
            println!("{:?}", cs.which_is_unsatisfied().unwrap());
        }
        assert!(cs.is_satisfied());
    }

    #[allow(dead_code)]
    pub(crate) fn frobenius_tests<
        FE: Field,
        ConstraintF: Field,
        F: FieldGadget<FE, ConstraintF>,
    >(
        maxpower: usize,
    ) {
        let mut cs = TestConstraintSystem::<ConstraintF>::new();
        let mut rng = XorShiftRng::seed_from_u64(1231275789u64);
        for i in 0..=maxpower {
            let mut a = FE::rand(&mut rng);
            let mut a_gadget = F::alloc(cs.ns(|| format!("a_gadget_{:?}", i)), || Ok(a)).unwrap();
            a_gadget = a_gadget
                .frobenius_map(cs.ns(|| format!("frob_map_{}", i)), i)
                .unwrap();
            a.frobenius_map(i);

            assert_eq!(a_gadget.get_value().unwrap(), a);
        }

        assert!(cs.is_satisfied());
    }
}
