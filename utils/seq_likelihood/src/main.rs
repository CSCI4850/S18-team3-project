/*
 * Author: Samuel Remedios
 * Purpose:
 * Calculates likelihood of a sequence occcuring using markov transition matrices.
 * Reads two strings from stdin, generates probabilies as it reads, then calculates
 * the likelihood of that string occuring.
 *
 * TODO:
 *  - Take input from stdin
 *  - properly parse and format input in the same way that Madison's word2vec does
 *
 */

use std::env;
use std::fs::File;
use std::io::Read;
use std::collections::HashMap;

fn main() {
    let args: Vec<String> = env::args().collect();
    let src_filename = &args[1];
    let gen_filename = &args[2];

    let mut src_file = File::open(src_filename).expect("File not found");
    let mut src_buffer = String::new();
    src_file
        .read_to_string(&mut src_buffer)
        .expect("Error reading file");
    let vec_src_str = vectorize_input_string(&src_buffer[..]);

    let mut gen_file = File::open(gen_filename).expect("File not found");
    let mut gen_buffer = String::new();
    gen_file
        .read_to_string(&mut gen_buffer)
        .expect("Error reading file");
    let vec_gen_str = vectorize_input_string(&gen_buffer[..]);

    let transitions = get_transitions(vec_src_str);
    println!(
        "Transitions: {:?}\n\
         Likelihood: {:?}",
        transitions,
        calc_likelihood(vec_gen_str, &transitions)
    );
}

// Calculates the transition matrix lazily
// Params:
//      input_string: sequence to calculate transition matrix for
// Returns:
//      transitions: Tranistion map of all possible following words in [0,1]
fn get_transitions(input_string: Vec<&str>) -> HashMap<&str, HashMap<&str, f64>> {
    let mut transitions = HashMap::new();
    let mut totals: HashMap<&str, f64> = HashMap::new();

    // use a sliding window of length 2
    for iter in input_string.windows(2) {
        let cur_word = iter[0];
        let next_word = iter[1];
        *transitions
            .entry(cur_word)
            .or_insert(HashMap::new())
            .entry(next_word)
            .or_insert(0_f64) += 1_f64;
        *totals.entry(cur_word).or_insert(0_f64) += 1_f64;
    }

    // Convert next words into probabilities
    for (cur_word, transitions) in transitions.iter_mut() {
        for (_next_word, val) in transitions.iter_mut() {
            *val /= totals[&*cur_word];
        }
    }

    // return transition probabilities
    transitions
}

fn calc_likelihood<'a>(
    input_string: Vec<&'a str>,
    transitions: &HashMap<&'a str, HashMap<&'a str, f64>>,
) -> f64 {
    let mut likelihood: f64 = 1_f64;
    for iter in input_string.windows(2) {
        let cur_word = iter[0];
        let next_word = iter[1];
        if transitions.contains_key(&cur_word) && transitions[&cur_word].contains_key(&next_word) {
            likelihood *= transitions[&cur_word][&next_word];
        }
    }

    likelihood
}

// Converts a string slice to a vector of &str for use
// Splits on spaces
fn vectorize_input_string(s: &str) -> Vec<&str> {
    s.split_whitespace().collect()
}
