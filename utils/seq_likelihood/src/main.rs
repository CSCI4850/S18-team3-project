/*
 * Author: Samuel Remedios
 * Purpose:
 * Calculates likelihood of a sequence occcuring using markov transition matrices.
 * Reads two strings from stdin, generates probabilies as it reads, then calculates
 * the likelihood of that string occuring.
 *
 * TODO:
 * - convert transitions into number probabilities
 *  -Optimize space constraints
 *
 */

use std::collections::HashMap;

fn main() {
    let input_string = vec!['A', 'B', 'B', 'C', 'B', 'A', 'D', 'D', 'A', 'B', 'A', 'D']
        .iter()
        .map(|c| c.to_string())
        .collect();

    let transitions = get_transition_likelihood(input_string);
    println!("{:?}", transitions);
}

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
