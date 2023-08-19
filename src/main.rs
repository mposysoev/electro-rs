use chemfiles::Frame;
use chemfiles::Trajectory;
use std::env;
use std::f64::consts::PI;
use std::fs::File;
use std::io::prelude::*;
use std::io::{BufRead, BufReader};

fn init_box_sizes_array(file_name: &str, num_atoms: usize) -> Vec<f64> {
    let mut boxes_len: Vec<f64> = Vec::new();
    let file = File::open(file_name).unwrap();
    let reader = BufReader::new(file);

    let mut current_line_number: usize = 0;
    let mut current_frame_number: usize = 0;

    for line in reader.lines() {
        if current_line_number == (num_atoms + 2) * current_frame_number + 1 {
            current_frame_number += 1;
            let numbers_from_line = line
                .unwrap()
                .split_whitespace()
                .map(|num_str| num_str.parse::<f64>())
                .filter_map(Result::ok)
                .collect::<Vec<f64>>();
            if let Some(&current_box_size) = numbers_from_line.last() {
                boxes_len.push(current_box_size);
            }
        }
        current_line_number += 1;
    }
    return boxes_len;
}

fn init_charges_array(file_name: &str) -> Vec<f64> {
    let mut charges: Vec<f64> = Vec::new();

    let file = File::open(file_name).unwrap();
    let reader = BufReader::new(file);

    for line in reader.lines() {
        if let Ok(line) = line {
            let parts: Vec<&str> = line.split_whitespace().collect();

            if let Ok(charge) = parts[6].parse::<f64>() {
                charges.push(charge);
            }
        }
    }

    return charges;
}

fn calculate_distance(a: &[f64; 3], b: &[f64; 3], box_size: f64) -> f64 {
    let mut dx = b[0] - a[0];
    let mut dy = b[1] - a[1];
    let mut dz = b[2] - a[2];

    if dx > box_size / 2.0 {
        dx = dx - box_size;
    }
    if dy > box_size / 2.0 {
        dy = dy - box_size;
    }
    if dz > box_size / 2.0 {
        dz = dz - box_size;
    }
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn build_distance_matrix(positions: &mut [[f64; 3]], box_size: f64) -> Vec<Vec<f64>> {
    let num_atoms = positions.len();
    let mut distance_matrix = vec![vec![0.0; num_atoms]; num_atoms];

    for i in 0..num_atoms {
        for j in (i + 1)..num_atoms {
            let distance = calculate_distance(&positions[i], &positions[j], box_size);
            distance_matrix[i][j] = distance;
            distance_matrix[j][i] = distance;
        }
    }

    distance_matrix
}

fn calculate_total_energy(matrix_distance: &Vec<Vec<f64>>, charge: &Vec<f64>) -> f64 {
    let n_mono = 67;
    let mut energy = 0.0;
    let mut diag_count = 0;

    for i in 0..matrix_distance[0].len() {
        for j in diag_count..matrix_distance[0].len() {
            if i == j {
                continue;
            } else {
                let r = matrix_distance[i][j];
                let q_1 = charge[i % n_mono];
                let q_2 = charge[j % n_mono];
                energy += q_1 * q_2 / r;
            }
        }
        diag_count += 1;
    }

    energy
}

fn calculate_energy_for_monomers(
    matrix_distance: &Vec<Vec<f64>>,
    charge: &Vec<f64>,
) -> (Vec<f64>, f64) {
    let n_mono = 67;
    let mut energies = vec![];
    let mut energy = 0.0;
    let mut diag_count = 0;
    let mut count_atoms_in_mono = 0;
    let mut count_mono_in_micelle = 1;

    for i in 0..matrix_distance[0].len() {
        for j in diag_count..(n_mono * count_mono_in_micelle) {
            if i == j {
                continue;
            } else {
                let r = matrix_distance[i][j];
                let q_1 = charge[i % n_mono];
                let q_2 = charge[j % n_mono];
                energy += q_1 * q_2 / r;
            }
        }

        diag_count += 1;
        count_atoms_in_mono += 1;
        if count_atoms_in_mono == n_mono {
            energies.push(energy);
            energy = 0.0;
            count_atoms_in_mono = 0;
            count_mono_in_micelle += 1;
        }
    }

    let all_monomers_energy = energies.iter().sum();
    (energies, all_monomers_energy)
}

fn main() {
    let TEMPERATURE: f64 = 223.0;
    let K_KULON_CONST: f64 = 1_f64 / (4_f64 * PI * 8.85418781762039_f64);
    let PEREVOD: f64 = 1.60217663_f64.powf(2.0) * 10_f64.powf(-16.0);
    let K_BOLTZMAN: f64 = 1.380649_f64 * 10_f64.powf(-23_f64);
    let UNIT_COEF = PEREVOD * K_KULON_CONST / K_BOLTZMAN / TEMPERATURE;

    let args: Vec<String> = env::args().collect();
    let charges_path = &args[1];
    let trj_file_path = &args[2];

    let mut trajectory = Trajectory::open(trj_file_path, 'r').unwrap();

    let mut frame = Frame::new();
    trajectory.read(&mut frame).unwrap();
    let atoms_num = frame.positions().len(); // amount of atoms

    let box_sizes = init_box_sizes_array(trj_file_path, atoms_num);
    let charges = init_charges_array(charges_path);

    let file_length = trajectory.nsteps(); // amount of frames

    let mut full_energy_array: Vec<f64> = Vec::with_capacity(file_length);
    let mut energy_benefit_array: Vec<f64> = Vec::with_capacity(file_length);
    let mut energies_of_mono_array: Vec<Vec<f64>> = Vec::with_capacity(file_length * atoms_num); // more than just length

    for i in 1..file_length {
        trajectory.read(&mut frame).unwrap();
        let mut positions: &mut [[f64; 3]] = frame.positions_mut();
        let (coms, mut positions) = positions.split_at_mut(1);
        let com_x = coms[0][0];
        let com_y = coms[0][1];
        let com_z = coms[0][2];

        for atom_i in &mut *positions {
            atom_i[0] -= com_x;
            atom_i[1] -= com_y;
            atom_i[2] -= com_z;
        }

        for atom_i in &mut *positions {
            if atom_i[0] > box_sizes[i] / 2.0 {
                atom_i[0] -= box_sizes[i];
            }
            if atom_i[1] > box_sizes[i] / 2.0 {
                atom_i[1] -= box_sizes[i];
            }
            if atom_i[2] > box_sizes[i] / 2.0 {
                atom_i[2] -= box_sizes[i];
            }
            if atom_i[0] < -box_sizes[i] / 2.0 {
                atom_i[0] += box_sizes[i];
            }
            if atom_i[1] < -box_sizes[i] / 2.0 {
                atom_i[1] += box_sizes[i];
            }
            if atom_i[2] < -box_sizes[i] / 2.0 {
                atom_i[2] += box_sizes[i];
            }
        }

        let matrix_distance = build_distance_matrix(positions, box_sizes[i]);

        let total_energy = calculate_total_energy(&matrix_distance, &charges);
        let (mut energies_of_mono, all_monomers_energy) =
            calculate_energy_for_monomers(&matrix_distance, &charges);
        let energy_benefit = total_energy - all_monomers_energy;

        full_energy_array.push(UNIT_COEF * total_energy);

        for element in energies_of_mono.iter_mut() {
            *element *= UNIT_COEF;
        }

        energies_of_mono_array.push(energies_of_mono);
        energy_benefit_array.push(UNIT_COEF * energy_benefit);
    }

    let mut file_for_full = File::create("full_energy.txt").expect("Failed to create file");
    for value in &full_energy_array {
        writeln!(file_for_full, "{}", value).expect("Failed to write to file");
    }

    let mut file_for_benefit = File::create("benefit_energy.txt").expect("Failed to create file");
    for value in &energy_benefit_array {
        writeln!(file_for_benefit, "{}", value).expect("Failed to write to file");
    }

    let mut file_for_monomers = File::create("monomers_energy.txt").expect("Failed to create file");
    for inner_vec in &energies_of_mono_array {
        for value in inner_vec {
            writeln!(file_for_monomers, "{}", value).expect("Failed to write to file");
        }
    }
}
