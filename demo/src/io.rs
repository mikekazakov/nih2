use nih::math::*;
use nih::render::*;
use std::path::Path;

pub fn load_obj<P: AsRef<Path>>(path: P) -> nih::render::MeshData {
    let obj_string = std::fs::read_to_string(path).unwrap();
    let model = wavefront_obj::obj::parse(obj_string).unwrap();
    let mut mesh = nih::render::MeshData::default();
    let geometries = model.objects[0].geometry.len();
    for geometry in 0..geometries {
        let start = mesh.positions.len();
        for prim in model.objects[0].geometry[geometry].shapes.iter() {
            match prim.primitive {
                wavefront_obj::obj::Primitive::Triangle(v0, v1, v2) => {
                    mesh.positions.push(Vec3::new(
                        model.objects[0].vertices[v0.0].x as f32,
                        model.objects[0].vertices[v0.0].y as f32,
                        model.objects[0].vertices[v0.0].z as f32,
                    ));
                    mesh.positions.push(Vec3::new(
                        model.objects[0].vertices[v1.0].x as f32,
                        model.objects[0].vertices[v1.0].y as f32,
                        model.objects[0].vertices[v1.0].z as f32,
                    ));
                    mesh.positions.push(Vec3::new(
                        model.objects[0].vertices[v2.0].x as f32,
                        model.objects[0].vertices[v2.0].y as f32,
                        model.objects[0].vertices[v2.0].z as f32,
                    ));
                    mesh.tex_coords.push(Vec2::new(
                        model.objects[0].tex_vertices[v0.1.unwrap()].u as f32,
                        model.objects[0].tex_vertices[v0.1.unwrap()].v as f32,
                    ));
                    mesh.tex_coords.push(Vec2::new(
                        model.objects[0].tex_vertices[v1.1.unwrap()].u as f32,
                        model.objects[0].tex_vertices[v1.1.unwrap()].v as f32,
                    ));
                    mesh.tex_coords.push(Vec2::new(
                        model.objects[0].tex_vertices[v2.1.unwrap()].u as f32,
                        model.objects[0].tex_vertices[v2.1.unwrap()].v as f32,
                    ));
                    mesh.normals.push(Vec3::new(
                        model.objects[0].normals[v0.2.unwrap()].x as f32,
                        model.objects[0].normals[v0.2.unwrap()].y as f32,
                        model.objects[0].normals[v0.2.unwrap()].z as f32,
                    ));
                    mesh.normals.push(Vec3::new(
                        model.objects[0].normals[v1.2.unwrap()].x as f32,
                        model.objects[0].normals[v1.2.unwrap()].y as f32,
                        model.objects[0].normals[v1.2.unwrap()].z as f32,
                    ));
                    mesh.normals.push(Vec3::new(
                        model.objects[0].normals[v2.2.unwrap()].x as f32,
                        model.objects[0].normals[v2.2.unwrap()].y as f32,
                        model.objects[0].normals[v2.2.unwrap()].z as f32,
                    ));
                }
                _ => {}
            }
        }
        let tris_count = (mesh.positions.len() - start) / 3;
        mesh.sections.push(MeshDataSection {
            start_index: start,
            num_triangles: tris_count,
            material_index: 0, // TODO: materials
        });
    }
    mesh.aabb = AABB::from_points(&mesh.positions);
    mesh
}
