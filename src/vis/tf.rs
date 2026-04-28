use egui::epaint::PathShape;
use egui::{pos2, Color32, Frame, Painter, Response, Sense, Shape, Stroke, Ui, Vec2};
use emath::{Pos2, Rect, RectTransform};
use std::sync::{Arc, Mutex};

#[derive(Debug)]
pub struct Histogram1d {
    pub total: u32,
    pub max: u32,
    pub values: Vec<u32>,
}

pub struct TransferFunctionWidget1d {
    point_list: Vec<Pos2>,
    point_list_sorted: Vec<Pos2>,
    line_stroke: Stroke,
    pub histogram_data: Arc<Mutex<Option<Histogram1d>>>,
}

impl TransferFunctionWidget1d {
    pub fn new() -> Self {
        let point_list = vec![
            pos2(0.0, 1.0),
            pos2(1.0, 0.0),
            pos2(0.25, 0.75),
            pos2(0.5, 0.5),
            pos2(0.75, 0.25),
        ];
        let mut point_list_sorted = point_list.clone();
        point_list_sorted
            .sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));

        TransferFunctionWidget1d {
            point_list,
            point_list_sorted,
            line_stroke: Stroke::new(1.0, Color32::RED),
            histogram_data: Arc::new(Mutex::new(None)),
        }
    }

    fn draw_histogram(&self, painter: &Painter, to_screen: RectTransform) {
        if let Some(histogram_data) = self.histogram_data.lock().unwrap().as_ref() {
            let shapes = histogram_data
                .values
                .iter()
                .enumerate()
                .map(|(i, b)| {
                    let rel = *b as f32 / histogram_data.max as f32;
                    Shape::rect_filled(
                        to_screen.transform_rect(Rect::from_min_max(
                            pos2(i as f32 / histogram_data.values.len() as f32, 1.0 - rel),
                            pos2((i + 1) as f32 / histogram_data.values.len() as f32, 1.0),
                        )),
                        0.0,
                        Color32::GREEN.linear_multiply(0.25),
                    )
                })
                .collect::<Vec<_>>();

            painter.extend(shapes);
        }
    }
    fn draw_control_points(&mut self, ui: &mut egui::Ui) -> Response {
        let (response, painter) =
            ui.allocate_painter(Vec2::new(ui.available_width(), 150.0), Sense::hover());

        let to_screen = emath::RectTransform::from_to(
            Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
            response.rect,
        );
        self.draw_histogram(&painter, to_screen);

        let canvas_size = response.rect.size();
        let num_points = self.point_list.len();

        let bg_response = ui.interact(response.rect, response.id.with(0), Sense::click());
        if bg_response.clicked() {
            self.point_list.push(
                to_screen
                    .inverse()
                    .transform_pos(bg_response.hover_pos().unwrap_or(pos2(0.0, 0.0))),
            );
        }

        let control_point_radius = 4.0;
        let control_point_shapes: Vec<Shape> = self
            .point_list
            .iter_mut()
            .enumerate()
            .map(|(i, point)| {
                let hover_size = Vec2::splat(control_point_radius * 4.0);

                let point_in_screen = to_screen.transform_pos(*point);
                let hover_rect = Rect::from_center_size(point_in_screen, hover_size);

                let point_id = response.id.with(i + 1);
                let point_response = ui.interact(
                    hover_rect,
                    point_id,
                    Sense::DRAG | Sense::HOVER | Sense::CLICK,
                );

                *point = to_screen
                    .from()
                    .clamp(*point + point_response.drag_delta() / canvas_size);

                // clamp edge values
                if i == 0 {
                    point.x = 0.0;
                } else if i == 1 {
                    point.x = 1.0;
                }

                let point_in_screen = to_screen.transform_pos(*point);
                let stroke = ui.style().interact(&response).fg_stroke;

                let radius = if point_response.hovered() {
                    control_point_radius * 2.0
                } else {
                    control_point_radius
                };

                if point_response.clicked_by(egui::PointerButton::Secondary) && i != 0 && i != 1 {
                    point.x = -1.0; // to be removed
                }

                Shape::circle_stroke(point_in_screen, radius, stroke)
            })
            .collect();

        self.point_list.retain(|x| x.x >= 0.0);

        self.point_list_sorted = self.point_list.clone();
        self.point_list_sorted
            .sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));

        let points_in_screen: Vec<Pos2> = self
            .point_list_sorted
            .iter()
            .map(|p| to_screen * (*p))
            .collect();
        painter.add(PathShape::line(points_in_screen, self.line_stroke));

        painter.extend(control_point_shapes);

        response
    }

    pub fn draw_ui(&mut self, ui: &mut Ui) {
        Frame::canvas(ui.style()).show(ui, |ui| self.draw_control_points(ui));
    }
}
