use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

/// `ProfileRecord` represents a single profiling entry with a label, timing statistics,
/// and potential child records for nested profiling scopes.
#[derive(Default)]
pub struct ProfileRecord {
    label: String,
    average: f64,
    samples: u32,
    min: f64,
    max: f64,
    children: Vec<Rc<RefCell<ProfileRecord>>>,
}

impl ProfileRecord {
    /// Create a new `ProfileRecord` with the specified label.
    pub fn new(label: &str) -> Self {
        Self { label: label.to_string(), average: 0.0, samples: 0, min: f64::MAX, max: f64::MIN, children: vec![] }
    }

    /// Get or create a child record with the given label. If a child with the label already exists,
    /// returns it; otherwise, creates a new one.
    pub fn child(&mut self, label: &str) -> Rc<RefCell<ProfileRecord>> {
        for child in &self.children {
            if child.borrow().label == label {
                return Rc::clone(child);
            }
        }
        let new_child = Rc::new(RefCell::new(ProfileRecord::new(label)));
        self.children.push(Rc::clone(&new_child));
        new_child
    }

    /// Update this record's statistics with a new duration sample (in milliseconds).
    pub fn commit(&mut self, duration: f64) {
        self.average = (self.average * self.samples as f64 + duration) / ((self.samples + 1) as f64);
        self.samples += 1;
        self.min = self.min.min(duration);
        self.max = self.max.max(duration);
    }

    /// Get a reference to the child records.
    pub fn children(&self) -> &[Rc<RefCell<ProfileRecord>>] {
        &self.children
    }
}

struct ProfilerInternals {
    root: Rc<RefCell<ProfileRecord>>,
    stack: Vec<Rc<RefCell<ProfileRecord>>>,
}

/// `Profiler` manages a tree of `ProfileRecord`s and a stack for tracking nested scopes.
pub struct Profiler {
    // root: Rc<RefCell<ProfileRecord>>,
    // stack: Vec<Rc<RefCell<ProfileRecord>>>,
    body: RefCell<ProfilerInternals>,
}

impl Profiler {
    /// Create a new `Profiler` with a root record.
    pub fn new() -> Self {
        let root = Rc::new(RefCell::new(ProfileRecord::new("frame")));
        Self { body: RefCell::new(ProfilerInternals { root: Rc::clone(&root), stack: vec![root] }) }
        // root: Rc::clone(&root), stack: vec![root]
    }

    /// Enter a profiling scope with the specified label.
    /// Pushes a new or existing child record onto the stack.
    pub fn enter(&self, label: &str) {
        let mut body = self.body.borrow_mut();
        let current = body.stack.last().unwrap();
        let child = current.borrow_mut().child(label);
        body.stack.push(child);
    }

    /// Exit the current profiling scope, updating its record with the measured duration (in ms).
    pub fn exit(&self, duration: f64) {
        let mut body = self.body.borrow_mut();
        let record = body.stack.pop().unwrap();
        record.borrow_mut().commit(duration);
    }

    /// Print the profiling report, showing average durations for all records in a tree format.
    pub fn print(&self) {
        fn print_records(records: &[Rc<RefCell<ProfileRecord>>], depth: usize) {
            for record in records {
                let r = record.borrow();
                let header = if depth > 0 {
                    format!("{:>width$}|- {}", "", r.label, width = (depth - 1) * 4)
                } else {
                    r.label.clone()
                };
                println!("{:<40.40} {:>7.2}ms", header, r.average);
                print_records(&r.children(), depth + 1);
            }
        }
        print_records(&[Rc::clone(&self.body.borrow().root)], 0);
    }

    /// Reset the profiler, clearing all records and statistics.
    pub fn reset(&self) {
        let mut body = self.body.borrow_mut();
        body.root = Rc::new(RefCell::new(ProfileRecord::new("frame")));
        body.stack = vec![Rc::clone(&body.root)];
    }
}

// to shut the fuck up borrow checker
unsafe impl Send for Profiler {}
unsafe impl Sync for Profiler {}

/// `ProfileScope` is an RAII guard that automatically enters and exits a profiling
/// scope, measuring the time spent in the scope and reporting it to the `Profiler`.
pub struct ProfileScope<'a> {
    // label: &'a str,
    start: Instant,
    profiler: &'a Profiler,
}

impl<'a> ProfileScope<'a> {
    /// Create a new `ProfileScope` with the given label and profiler.
    /// Enters the profiling scope on creation.
    pub fn new(label: &'a str, profiler: &'a Profiler) -> Self {
        profiler.enter(label);
        Self { start: Instant::now(), profiler }
    }
}

impl<'a> Drop for ProfileScope<'a> {
    /// On drop, exits the profiling scope and records the elapsed time.
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        self.profiler.exit(duration.as_secs_f64() * 1000.0);
    }
}

// // Example usage
// fn main() {
//     let profiler = RefCell::new(Profiler::new());
//
//     {
//         let _scope = ProfileScope::new("main_task", &profiler);
//         std::thread::sleep(Duration::from_millis(10));
//     }
//
//     profiler.borrow().print();
// }

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn test_profile_record_new() {
        let rec = ProfileRecord::new("test_label");
        assert_eq!(rec.label, "test_label");
        assert_eq!(rec.average, 0.0);
        assert_eq!(rec.samples, 0);
        assert_eq!(rec.min, f64::MAX);
        assert_eq!(rec.max, f64::MIN);
        assert!(rec.children.is_empty());
    }

    #[test]
    fn test_profile_record_child() {
        let mut rec = ProfileRecord::new("parent");
        let child1 = rec.child("child1");
        assert_eq!(child1.borrow().label, "child1");
        assert_eq!(rec.children.len(), 1);
        // Getting the same child returns the same Rc
        let child1_again = rec.child("child1");
        assert!(Rc::ptr_eq(&child1, &child1_again));
        // Adding a new child
        let child2 = rec.child("child2");
        assert_eq!(child2.borrow().label, "child2");
        assert_eq!(rec.children.len(), 2);
    }

    #[test]
    fn test_profile_record_commit_and_stats() {
        let mut rec = ProfileRecord::new("stats");
        rec.commit(10.0);
        assert_eq!(rec.samples, 1);
        assert_eq!(rec.average, 10.0);
        assert_eq!(rec.min, 10.0);
        assert_eq!(rec.max, 10.0);
        rec.commit(20.0);
        assert_eq!(rec.samples, 2);
        assert_eq!(rec.average, 15.0);
        assert_eq!(rec.min, 10.0);
        assert_eq!(rec.max, 20.0);
        rec.commit(5.0);
        assert_eq!(rec.samples, 3);
        assert!((rec.average - 11.666666).abs() < 1e-5);
        assert_eq!(rec.min, 5.0);
        assert_eq!(rec.max, 20.0);
    }

    #[test]
    fn test_profiler_new_root() {
        let profiler = Profiler::new();
        {
            let body = profiler.body.borrow();
            let root = body.root.borrow();
            assert_eq!(root.label, "frame");
            assert_eq!(root.samples, 0);
            assert_eq!(root.average, 0.0);
            assert!(root.children.is_empty());
        }
        assert_eq!(profiler.body.borrow().stack.len(), 1);
        assert!(Rc::ptr_eq(&profiler.body.borrow().root, &profiler.body.borrow().stack[0]));
    }

    #[test]
    fn test_profiler_enter_exit() {
        let profiler = Profiler::new();

        profiler.enter("scope1");
        assert_eq!(profiler.body.borrow().stack.len(), 2);
        {
            let body = profiler.body.borrow();
            let top = body.stack.last().unwrap();
            assert_eq!(top.borrow().label, "scope1");
            assert_eq!(top.borrow().samples, 0);
        }

        profiler.exit(10.0);
        assert_eq!(profiler.body.borrow().stack.len(), 1);

        {
            let body = profiler.body.borrow();
            let root = body.root.borrow();
            assert_eq!(root.children.len(), 1);
            let child = &root.children[0];
            let child_borrow = child.borrow();
            assert_eq!(child_borrow.label, "scope1");
            assert_eq!(child_borrow.samples, 1);
            assert_eq!(child_borrow.average, 10.0);
            assert_eq!(child_borrow.min, 10.0);
            assert_eq!(child_borrow.max, 10.0);
        }
    }

    #[test]
    fn test_profiler_reset() {
        let profiler = Profiler::new();
        profiler.enter("scope1");
        profiler.exit(5.0);
        assert_eq!(profiler.body.borrow().root.borrow().children.len(), 1);

        profiler.reset();
        {
            let body = profiler.body.borrow();
            let root = body.root.borrow();
            assert_eq!(root.label, "frame");
            assert_eq!(root.samples, 0);
            assert!(root.children.is_empty());
            assert_eq!(body.stack.len(), 1);
            assert!(Rc::ptr_eq(&body.root, &body.stack[0]));
        }
    }

    #[test]
    fn test_profiler_nested_scopes_and_averages() {
        let profiler = Profiler::new();

        // Enter first scope
        profiler.enter("outer");
        // Simulate commit of 20 ms
        {
            let body = profiler.body.borrow();
            let current = body.stack.last().unwrap();
            current.borrow_mut().commit(20.0);
        }

        // Enter nested scope
        profiler.enter("inner");
        // Simulate commit of 10 ms
        {
            let body = profiler.body.borrow();
            let current = body.stack.last().unwrap();
            current.borrow_mut().commit(10.0);
        }
        // Exit inner scope
        profiler.exit(10.0);

        // Commit another sample for outer scope
        {
            let body = profiler.body.borrow();
            let current = body.stack.last().unwrap();
            current.borrow_mut().commit(30.0);
        }

        // Exit outer scope
        profiler.exit(25.0);

        let body = profiler.body.borrow();
        let root = body.root.borrow();
        assert_eq!(root.children.len(), 1);
        let outer = root.children[0].borrow();
        assert_eq!(outer.label, "outer");
        assert_eq!(outer.samples, 3); // two commits + one exit commit
        // Average should be (20 + 30 + 25) / 3 = 25.0
        assert!((outer.average - 25.0).abs() < 1e-5);

        assert_eq!(outer.children.len(), 1);
        let inner = outer.children[0].borrow();
        assert_eq!(inner.label, "inner");
        assert_eq!(inner.samples, 2); // one commit + one exit commit
        assert!((inner.average - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_profile_scope_creation_and_stack_update() {
        let profiler = Profiler::new();

        {
            let _scope = ProfileScope::new("test_scope", &profiler);
            assert_eq!(profiler.body.borrow().stack.len(), 2);
            let top_label = profiler.body.borrow().stack.last().unwrap().borrow().label.clone();
            assert_eq!(top_label, "test_scope");
        }

        // After _scope is dropped, stack should be back to root
        assert_eq!(profiler.body.borrow().stack.len(), 1);
        assert_eq!(profiler.body.borrow().stack[0].borrow().label, "frame");
    }

    #[test]
    fn test_profile_scope_drop_records_duration() {
        let profiler = Profiler::new();

        {
            let _scope = ProfileScope::new("timed_scope", &profiler);
            sleep(Duration::from_millis(20));
            // scope dropped here
        }

        let body = profiler.body.borrow();
        let root = body.root.borrow();
        assert_eq!(root.children.len(), 1);
        let child = &root.children[0];
        let child_borrow = child.borrow();
        assert_eq!(child_borrow.label, "timed_scope");
        assert_eq!(child_borrow.samples, 1);
        assert!(child_borrow.average >= 20.0);
        assert!(child_borrow.min >= 20.0);
        assert!(child_borrow.max >= 20.0);
    }

    #[test]
    fn test_profile_scope_nested_usage() {
        let profiler = Profiler::new();

        {
            let _outer = ProfileScope::new("outer_scope", &profiler);
            sleep(Duration::from_millis(10));

            {
                let _inner = ProfileScope::new("inner_scope", &profiler);
                sleep(Duration::from_millis(15));
            }

            sleep(Duration::from_millis(5));
        }

        let body = profiler.body.borrow();
        let root = body.root.borrow();
        assert_eq!(root.children.len(), 1);
        let outer = root.children[0].borrow();
        assert_eq!(outer.label, "outer_scope");
        assert_eq!(outer.children.len(), 1);

        let inner = outer.children[0].borrow();
        assert_eq!(inner.label, "inner_scope");
        assert_eq!(inner.samples, 1);
        assert!(inner.average >= 15.0);

        assert_eq!(outer.samples, 1);
        // outer scope duration should be at least 10 + 15 + 5 = 30 ms
        assert!(outer.average >= 30.0);
    }
}
