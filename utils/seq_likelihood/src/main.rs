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

    let transitions = get_transitions(stringvec);
    println!("{:?}", transitions);

    let test_string = "A B C B";
    let test_vec = vectorize_input_string(test_string);
    println!("{:?}", calc_likelihood(test_vec, &transitions));

    let test_string = "C B";
    let test_vec = vectorize_input_string(test_string);
    println!("{:?}", calc_likelihood(test_vec, &transitions));
}

// Calculates the transition matrix lazily
// Params:
//      input_string: sequence to calculate transition matrix for
// Returns:
//      transitions: Tranistion map of all possible following words in [0,1]
fn get_transitions(input_string: Vec<&str>) -> HashMap<&str, HashMap<&str, f32>> {
    let mut transitions = HashMap::new();
    let mut totals: HashMap<&str, f32> = HashMap::new();

    // use a sliding window of length 2
    for iter in input_string.windows(2) {
        let cur_word = iter[0];
        let next_word = iter[1];
        *transitions
            .entry(cur_word)
            .or_insert(HashMap::new())
            .entry(next_word)
            .or_insert(0_f32) += 1_f32;
        *totals.entry(cur_word).or_insert(0_f32) += 1_f32;
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
    transitions: &HashMap<&'a str, HashMap<&'a str, f32>>,
) -> f32 {
    let mut likelihood: f32 = 1_f32;
    for iter in input_string.windows(2) {
        let cur_word = iter[0];
        let next_word = iter[1];
        if transitions.contains_key(&cur_word) && transitions.contains_key(&next_word) {
            likelihood *= transitions[&cur_word][&next_word];
        }
    }

    likelihood
}

// Converts a string slice to a vector of &str for use
// Splits on spaces
fn vectorize_input_string(s: &str) -> Vec<&str> {
    s.split(" ").collect()
}
