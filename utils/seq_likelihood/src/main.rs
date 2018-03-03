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
    let transitions = vec!['A', 'B', 'B', 'C', 'B', 'A', 'D', 'D', 'A', 'B', 'A', 'D'];

    let mut hashmap = HashMap::new();
    let num_elements = transitions.len();
    for i in 0..num_elements - 1 {
        hashmap
            .entry(transitions[i].clone())
            .or_insert(Vec::new())
            .push(transitions[i + 1].clone());
    }

    println!("{:?}", hashmap);
}
