use metastasa::segment::{Segment, process_segment};
use metastasa::logic_attention::logical_attention;
use metastasa::segment::KnowledgeNode;
use uuid::Uuid;
use rand::Rng;

fn random_string(len: usize) -> String {
    let charset = b"абвгдеёжзийклмнопрстуфхцчшщъыьэюяABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789{}[]";
    let mut rng = rand::thread_rng();
    (0..len).map(|_| charset[rng.gen_range(0..charset.len())] as char).collect()
}

fn random_segment(depth: usize) -> Segment {
    if depth > 5 || rand::random::<f32>() < 0.5 {
        Segment::Primitive(random_string(10))
    } else {
        let n = rand::thread_rng().gen_range(1..4);
        Segment::Composite((0..n).map(|_| random_segment(depth + 1)).collect())
    }
}

#[test]
fn fuzz_process_segment() {
    for _ in 0..100 {
        let seg = random_segment(0);
        let _ = process_segment(&seg, 0); // Не должно паниковать
    }
}

#[test]
fn fuzz_logical_attention() {
    for _ in 0..100 {
        let seg = random_segment(0);
        let node = KnowledgeNode {
            id: Uuid::new_v4(),
            data: random_segment(0),
            depth: 0,
            edges: vec![],
            tags: vec![random_string(5)],
        };
        let _ = logical_attention(&seg, &[node]);
    }
} 