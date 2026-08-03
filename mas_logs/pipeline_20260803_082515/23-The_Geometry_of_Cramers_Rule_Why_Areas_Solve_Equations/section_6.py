from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section6Scene(TeachingScene):
    def construct(self):
        # Initial Setup
        title = "Summary and Real-World Intuition"
        lines = [
            "Solve equations by comparing geometric areas.",
            "The rule fails if the base area is zero.",
            "Geometry reveals the logic behind the algebra."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Grid Coordinates for Visualization
        # We'll use a small coordinate system in the right area
        # Origin at D2
        origin = self.grid["D2"]
        scale_map = 0.25
        
        def to_scene_coords(vec):
            return origin + np.array([vec[0]*scale_map, vec[1]*scale_map, 0])

        # Vectors: v1=[2,1], v2=[1,3], b=[7,11], solution x=2, y=3
        v1_raw = np.array([2, 1, 0])
        v2_raw = np.array([1, 3, 0])
        b_vec_raw = np.array([7, 11, 0])
        
        # Elements
        axes_x = Arrow(to_scene_coords([-1, 0]), to_scene_coords([8, 0]), color=GRAY, stroke_width=2, buff=0)
        axes_y = Arrow(to_scene_coords([0, -1]), to_scene_coords([0, 12]), color=GRAY, stroke_width=2, buff=0)
        
        target_b = Dot(to_scene_coords(b_vec_raw), color="#00FF00")
        label_b = MathTex("b", color="#00FF00", font_size=24).next_to(target_b, UR, buff=0.1)
        
        # Robot (represented as a circle with inner dot)
        robot = VGroup(
            Circle(radius=0.12, color=WHITE, fill_opacity=0.5),
            Dot(radius=0.04, color=WHITE)
        ).move_to(to_scene_coords([0,0]))
        
        self.play(Create(axes_x), Create(axes_y))
        self.play(FadeIn(target_b), Write(label_b))
        self.play(FadeIn(robot))
        
        # Move robot along x*v1 and y*v2
        path_x = Line(to_scene_coords([0,0]), to_scene_coords(2*v1_raw), color="#FF00FF")
        path_y = Line(to_scene_coords(2*v1_raw), to_scene_coords(b_vec_raw), color="#00FFFF")
        
        self.play(robot.animate.move_to(to_scene_coords(2*v1_raw)), Create(path_x), run_time=1.5)
        self.play(robot.animate.move_to(to_scene_coords(b_vec_raw)), Create(path_y), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Clear path, move robot back, then clear them
        self.play(FadeOut(path_x), FadeOut(path_y), robot.animate.move_to(to_scene_coords([0,0])))
        self.play(FadeOut(target_b), FadeOut(label_b), FadeOut(robot))
        
        # Show Determinant Collapsing
        # Parallelogram points
        p_points = [
            to_scene_coords([0,0]),
            to_scene_coords(v1_raw),
            to_scene_coords(v1_raw + v2_raw),
            to_scene_coords(v2_raw)
        ]
        parallelogram = Polygon(*p_points, color=BLUE, fill_opacity=0.3)
        
        # Vectors as visual aids
        vec1 = Vector(to_scene_coords(v1_raw) - origin, color="#FF00FF").shift(origin)
        vec2 = Vector(to_scene_coords(v2_raw) - origin, color="#00FFFF").shift(origin)

        # Labels for the vectors
        v1_label = MathTex("\\vec{v}_1", color="#FF00FF", font_size=20).next_to(vec1.get_end(), DR, buff=0.1)
        v2_label = MathTex("\\vec{v}_2", color="#00FFFF", font_size=20).next_to(vec2.get_end(), UL, buff=0.1)

        # Det label - Resolve Issue 43 (position C4)
        det_text = MathTex("det(A) \\neq 0", color=WHITE, font_size=28)
        self.place_at_grid(det_text, "C4")

        self.play(Create(vec1), Create(vec2), Create(v1_label), Create(v2_label), FadeIn(parallelogram))
        self.play(Write(det_text))
        self.wait(0.5)
        
        # Collapse: make v2 parallel to v1
        v2_new_raw = v1_raw * 2.0
        
        new_vec2_end = to_scene_coords(v2_new_raw)
        new_p_points = [
            to_scene_coords([0,0]),
            to_scene_coords(v1_raw),
            to_scene_coords(v1_raw + v2_new_raw),
            to_scene_coords(v2_new_raw)
        ]
        
        # Resolve Issue 43 (position C4)
        det_zero = MathTex("det(A) = 0", color=RED, font_size=28)
        self.place_at_grid(det_zero, "C4")

        self.play(
            vec2.animate.put_start_and_end_on(origin, new_vec2_end),
            v2_label.animate.next_to(new_vec2_end, UR, buff=0.1),
            parallelogram.animate.set_points_as_corners(new_p_points),
            Transform(det_text, det_zero),
            run_time=2
        )
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Fade out geometry
        self.play(
            FadeOut(vec1), FadeOut(vec2), FadeOut(v1_label), FadeOut(v2_label),
            FadeOut(parallelogram), FadeOut(det_text), 
            FadeOut(axes_x), FadeOut(axes_y)
        )
        
        # Show final solved coordinates - Resolve Issue 42 (D2-F5)
        final_coords = MathTex("(x, y) = (2, 3)", color=WHITE, font_size=36)
        self.place_in_area(final_coords, "D2", "F5")
        
        self.play(Write(final_coords))
        self.wait(2)
        
        # Final cleanup
        self.play(FadeOut(final_coords), FadeOut(self.title), FadeOut(self.lecture))
        self.wait(1)
