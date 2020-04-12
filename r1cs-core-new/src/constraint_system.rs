use crate::{
    BTreeMap, LcIndex, LinearCombination, Matrix, Rc, String, SynthesisError, Variable, Vec,
};
use algebra_core::Field;
use core::cell::{Ref, RefCell, RefMut};
use core::ops::Deref;

/// Computations are expressed in terms of rank-1 constraint systems (R1CS).
/// The `generate_constraints` method is called to generate constraints for
/// both CRS generation and for proving.
///
/// TODO: Think: should we replace this with just a closure?
pub trait ConstraintSynthesizer<F: Field> {
    /// Drives generation of new constraints inside `CS`.
    fn generate_constraints(self, cs: &mut ConstraintSystem<F>) -> Result<(), SynthesisError>;
}

/// An Rank-One `ConstraintSystem`. Enforces constraints of the form
/// `⟨a_i, z⟩ ⋅ ⟨b_i, z⟩ = ⟨c_i, z⟩`, where `a_i`, `b_i`, and `c_i` are linear
/// combinations over variables, and `z` is the concrete assignment to these
/// variables.
pub struct ConstraintSystem<F: Field> {
    /// The mode in which the constraint system is operating. `self` can either
    /// be in setup mode (i.e., `self.mode == Mode::Setup`) or in proving mode
    /// (i.e., `self.mode == Mode::Prove`). If we are in proving mode, then we
    /// have the additional option of whether or not to construct the A, B, and
    /// C matrices of the constraint system (see below).
    pub mode: Mode,
    /// The number of variables that are "public inputs" to the constraint system.
    pub num_instance_variables: usize,
    /// The number of variables that are "private inputs" to the constraint system.
    pub num_witness_variables: usize,
    /// The number of constraints in the constraint system.
    pub num_constraints: usize,
    /// The number of linear combinations
    pub num_linear_combinations: usize,

    /// Assignments to the public input variables. This is empty if `self.mode == Mode::Setup`.
    pub instance_assignment: Vec<F>,
    /// Assignments to the private input variables. This is empty if `self.mode == Mode::Setup`.
    pub witness_assignment: Vec<F>,

    lc_map: BTreeMap<LcIndex, LinearCombination<F>>,
    namespace: Vec<String>,
    current_namespace_path: String,
    constraint_names: Vec<String>,

    a_constraints: Vec<LcIndex>,
    b_constraints: Vec<LcIndex>,
    c_constraints: Vec<LcIndex>,
}

/// Defines the mode of operation of a `ConstraintSystem`.
#[derive(Eq, PartialEq, Ord, PartialOrd)]
pub enum Mode {
    /// Indicate to the `ConstraintSystem` that it should only generate
    /// constraint matrices and not populate the variable assignments.
    Setup,
    /// Indicate to the `ConstraintSystem` that it populate the variable
    /// assignments. If additionally `construct_matrices == true`, then generate
    /// the matrices as in the `Setup` case.
    Prove {
        /// If `construct_matrices == true`, then generate
        /// the matrices as in the `Setup` case.
        construct_matrices: bool,
    },
}

impl<F: Field> ConstraintSystem<F> {
    #[inline]
    fn make_row(l: &LinearCombination<F>) -> Vec<(F, usize)> {
        l.0.iter()
            .filter_map(|(coeff, var)| {
                if coeff.is_zero() {
                    None
                } else {
                    Some((*coeff, var.get_index_unchecked().expect("no symbolic LCs")))
                }
            })
            .collect()
    }

    /// Construct an ampty `StandardConstraintSystem`.
    pub fn new() -> Self {
        Self {
            num_instance_variables: 1,
            num_witness_variables: 0,
            num_constraints: 0,
            num_linear_combinations: 0,
            a_constraints: Vec::new(),
            b_constraints: Vec::new(),
            c_constraints: Vec::new(),
            instance_assignment: Vec::new(),
            witness_assignment: Vec::new(),

            constraint_names: Vec::new(),
            namespace: Vec::new(),
            current_namespace_path: String::new(),
            lc_map: BTreeMap::new(),

            mode: Mode::Setup,
        }
    }

    /// Check whether `self.mode == Mode::Setup`.
    pub fn is_in_setup_mode(&self) -> bool {
        self.mode == Mode::Setup
    }

    /// Check whether or not `self` will construct matrices.
    pub fn should_construct_matrices(&self) -> bool {
        match self.mode {
            Mode::Setup => true,
            Mode::Prove { construct_matrices } => construct_matrices,
        }
    }

    #[inline]
    fn compute_full_name(&self, name: impl Into<String>) -> String {
        [self.current_namespace_path.clone(), name.into()].join("/")
    }

    /// Return a variable representing the constant "zero" inside the constraint
    /// system.
    #[inline]
    pub fn zero() -> Variable {
        Variable::Zero
    }

    /// Return a variable representing the constant "one" inside the constraint
    /// system.
    #[inline]
    pub fn one() -> Variable {
        Variable::One
    }

    /// Obtain a variable representing a new public instance input.
    #[inline]
    pub fn new_instance_variable<Func>(&mut self, f: Func) -> Result<Variable, SynthesisError>
    where
        Func: FnOnce() -> Result<F, SynthesisError>,
    {
        let index = self.num_instance_variables;
        self.num_instance_variables += 1;

        if !self.is_in_setup_mode() {
            self.instance_assignment.push(f()?);
        }
        Ok(Variable::Instance(index))
    }

    /// Obtain a variable representing a new private witness input.
    #[inline]
    pub fn new_witness_variable<Func>(&mut self, f: Func) -> Result<Variable, SynthesisError>
    where
        Func: FnOnce() -> Result<F, SynthesisError>,
    {
        let index = self.num_witness_variables;
        self.num_witness_variables += 1;

        if !self.is_in_setup_mode() {
            self.witness_assignment.push(f()?);
        }
        Ok(Variable::Witness(index))
    }

    /// Obtain a variable representing a linear combination.
    #[inline]
    pub fn new_lc(&mut self, lc: LinearCombination<F>) -> Result<Variable, SynthesisError> {
        let index = LcIndex(self.num_linear_combinations);
        let var = Variable::SymbolicLc(index);

        self.lc_map.insert(index, lc);

        self.num_linear_combinations += 1;
        Ok(var)
    }

    /// Enforce a R1CS constraint with an automatically generated name.
    #[inline]
    pub fn enforce_constraint(
        &mut self,
        a: LinearCombination<F>,
        b: LinearCombination<F>,
        c: LinearCombination<F>,
    ) -> Result<(), SynthesisError> {
        let name = crate::format!("{}", self.num_constraints);
        self.enforce_named_constraint(a, b, c, name)
    }

    /// Enforce a R1CS constraint with the name `name`.
    #[inline]
    pub fn enforce_named_constraint(
        &mut self,
        a: LinearCombination<F>,
        b: LinearCombination<F>,
        c: LinearCombination<F>,
        name: String,
    ) -> Result<(), SynthesisError> {
        if self.should_construct_matrices() {
            let a_index = self.new_lc(a)?.get_lc_index().unwrap();
            let b_index = self.new_lc(b)?.get_lc_index().unwrap();
            let c_index = self.new_lc(c)?.get_lc_index().unwrap();
            self.a_constraints.push(a_index);
            self.b_constraints.push(b_index);
            self.c_constraints.push(c_index);
        }
        self.num_constraints += 1;
        let name = self.compute_full_name(name);
        self.constraint_names.push(name);
        Ok(())
    }

    /// Enter a new namespace.
    #[inline]
    pub fn enter_namespace(&mut self, name: String) {
        self.namespace.push(name);
        self.current_namespace_path = self.namespace.join("/");
    }

    /// Leave a namespace.
    #[inline]
    pub fn leave_namespace(&mut self) {
        self.namespace.pop();
        self.current_namespace_path = self.namespace.join("/");
    }

    /// Naively inlines symbolic linear combinations into the linear combinations
    /// that use them.
    ///
    /// Useful for standard pairing-based SNARKs where addition gates are free.
    pub fn inline_all_lcs(&mut self) {
        let mut inlined_lcs = BTreeMap::new();
        for (&index, lc) in &self.lc_map {
            let mut inlined_lc = LinearCombination::new();
            for &(coeff, var) in lc.iter() {
                if var.is_lc() {
                    let lc_index = var.get_lc_index().expect("should be lc");
                    // If `var` is a `SymbolicLc`, fetch the corresponding
                    // inlined LC, and substitute it in.
                    let lc = inlined_lcs.get(&lc_index).expect("should be inlined");
                    inlined_lc.extend((lc * coeff).0.into_iter());
                } else {
                    // Otherwise, it's a concrete variable and so we
                    // substitute it in directly.
                    inlined_lc.push((coeff, var));
                }
            }
            inlined_lc.compactify();
            inlined_lcs.insert(index, inlined_lc);
        }
        self.lc_map = inlined_lcs;
    }

    /// If a `SymbolicLc` is used in more than one location, this method makes a new
    /// variable for that `SymbolicLc`, adds a constraint ensuring the equality of
    /// the variable and the linear combination, and then uses that variable in every
    /// location the `SymbolicLc` is used.
    ///
    /// Useful for SNARKs like `Marlin` or `Fractal`, where where addition gates
    /// are not (entirely) free.
    pub fn outline_lcs(&mut self) {
        unimplemented!()
    }

    /// This step must be called after constraint generation has completed, and after
    /// all symbolic LCs have been inlined into the places that they are used.
    #[inline]
    pub fn into_matrices(self) -> Option<ConstraintMatrices<F>> {
        if let Mode::Prove {
            construct_matrices: false,
        } = self.mode
        {
            let a: Vec<_> = self
                .a_constraints
                .iter()
                .map(|index| Self::make_row(self.lc_map.get(index).unwrap()))
                .collect();
            let b: Vec<_> = self
                .b_constraints
                .iter()
                .map(|index| Self::make_row(self.lc_map.get(index).unwrap()))
                .collect();
            let c: Vec<_> = self
                .c_constraints
                .iter()
                .map(|index| Self::make_row(self.lc_map.get(index).unwrap()))
                .collect();

            let a_num_non_zero: usize = a.iter().map(|lc| lc.len()).sum();
            let b_num_non_zero: usize = b.iter().map(|lc| lc.len()).sum();
            let c_num_non_zero: usize = c.iter().map(|lc| lc.len()).sum();
            Some(ConstraintMatrices {
                num_instance_variables: self.num_instance_variables,
                num_witness_variables: self.num_witness_variables,
                num_constraints: self.num_constraints,

                a_num_non_zero,
                b_num_non_zero,
                c_num_non_zero,

                a,
                b,
                c,
            })
        } else {
            None
        }
    }
}

pub struct ConstraintMatrices<F: Field> {
    /// The number of variables that are "public instances" to the constraint system.
    pub num_instance_variables: usize,
    /// The number of variables that are "private witnesses" to the constraint system.
    pub num_witness_variables: usize,
    /// The number of constraints in the constraint system.
    pub num_constraints: usize,
    /// The number of non_zero entries in the A matrix.
    pub a_num_non_zero: usize,
    /// The number of non_zero entries in the B matrix.
    pub b_num_non_zero: usize,
    /// The number of non_zero entries in the C matrix.
    pub c_num_non_zero: usize,

    /// The A constraint matrix. This is `None` when
    /// `self.mode == Mode::Prove { construct_matrices = false }`.
    pub a: Matrix<F>,
    /// The B constraint matrix. This is `None` when
    /// `self.mode == Mode::Prove { construct_matrices = false }`.
    pub b: Matrix<F>,
    /// The C constraint matrix. This is `None` when
    /// `self.mode == Mode::Prove { construct_matrices = false }`.
    pub c: Matrix<F>,
}

/// A shared reference to a constraint system that can be stored in high level
/// variables.
#[derive(Clone)]
pub struct ConstraintSystemRef<F: Field> {
    inner: Rc<RefCell<ConstraintSystem<F>>>,
}

impl<F: Field> ConstraintSystemRef<F> {
    /// Construct a `ConstraintSystemRef` from a `ConstraintSystem`.
    #[inline]
    pub fn new(inner: ConstraintSystem<F>) -> Self {
        Self {
            inner: Rc::new(RefCell::new(inner)),
        }
    }

    /// Obtain an immutable reference to the underlying `ConstraintSystem`.
    ///
    /// # Panics
    /// This method panics if `self` is already mutably borrowed.
    #[inline]
    pub fn borrow(&self) -> Ref<ConstraintSystem<F>> {
        self.inner.borrow()
    }

    /// Obtain a mutable reference to the underlying `ConstraintSystem`.
    ///
    /// # Panics
    /// This method panics if `self` is already mutably borrowed.
    #[inline]
    pub fn borrow_mut(&self) -> RefMut<ConstraintSystem<F>> {
        self.inner.borrow_mut()
    }

    /// Obtain a raw mutable reference to the underlying `ConstraintSystem`.
    #[inline]
    pub fn as_ptr(&self) -> *mut ConstraintSystem<F> {
        self.inner.as_ptr()
    }
}

impl<'a, F: Field> Deref for ConstraintSystemRef<F> {
    type Target = ConstraintSystem<F>;

    /// Obtain an immutable reference to the underlying `ConstraintSystem`.
    ///
    /// # Panics
    /// This method panics if `self` is already mutably borrowed.
    #[inline]
    #[allow(unsafe_code)]
    fn deref(&self) -> &ConstraintSystem<F> {
        unsafe { self.as_ptr().as_ref().unwrap() }
    }
}
