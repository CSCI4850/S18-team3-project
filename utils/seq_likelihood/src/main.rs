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

use std::collections::HashMap;

fn main() {
    let input_string = "A B B C B A D D A B A D";
    let stringvec = vectorize_input_string(input_string);
    println!("{:?}", stringvec);

    let transitions = get_transition_likelihood(stringvec);
    println!("{:?}", transitions);
}

// Calculates the transition matrix lazily
// Params:
//      input_string: sequence to calculate transition matrix for
// Returns:
//      transitions: Likelihood of all possible following words in [0,1]
fn get_transition_likelihood(input_string: Vec<String>) -> HashMap<String, HashMap<String, f32>> {
    let mut transitions = HashMap::new();
    let mut totals = HashMap::new();
    let num_elements = input_string.len();
    for i in 0..num_elements - 1 {
        *transitions
            .entry(input_string[i].clone())
            .or_insert(HashMap::new())
            .entry(input_string[i + 1].clone())
            .or_insert(0_f32) += 1_f32;
        *totals.entry(input_string[i].clone()).or_insert(0_f32) += 1_f32;
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

// Converts a string slice to a vector of Strings for use
// Splits on spaces
fn vectorize_input_string(s: &str) -> Vec<String> {
    s.split(" ")
        .map(|i| i.parse::<String>().expect("Error reading input."))
        .collect()
}
