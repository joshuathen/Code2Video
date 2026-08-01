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
        # Setup layout
        lecture_lines = [
            "Cramer's rule is a geometric ratio of areas.",
            "It measures how b contributes to the space.",
            "The formula works for any number of dimensions.",
            "Our delivery drone has found its precise path.",
            "Linear algebra solves problems with geometry."
        ]
        self.setup_layout("Summary and Intuition", lecture_lines)

        # Coordinate conversion for the visual grid
        # Origin at E2 (1.5, -1.8), scale factor 0.5
        origin_pos = self.grid["E2"]
        coord_scale = 0.5
        
        def to_screen(point):
            return origin_pos + np.array([point[0]*coord_scale, point[1]*coord_scale, 0])

        # Objects for the geometry
        v1_raw = np.array([2, 1, 0])
        v2_raw = np.array([1, 2, 0])
        b_raw = np.array([7, 6, 0])
        
        v1_vec = Arrow(origin_pos, to_screen(v1_raw), buff=0, color="#00FF00")
        v2_vec = Arrow(origin_pos, to_screen(v2_raw), buff=0, color="#0000FF")
        b_vec = Arrow(origin_pos, to_screen(b_raw), buff=0, color="#FFFF00")
        
        v1_label = MathTex("v_1", color="#00FF00", font_size=20).next_to(to_screen(v1_raw), RIGHT, buff=0.1)
        v2_label = MathTex("v_2", color="#0000FF", font_size=20).next_to(to_screen(v2_raw), LEFT, buff=0.1)
        b_label = MathTex("b", color="#FFFF00", font_size=20).next_to(to_screen(b_raw), UP, buff=0.1)

        # Parallelogram for det(v1, v2)
        poly_a = Polygon(
            origin_pos, to_screen(v1_raw), to_screen(v1_raw + v2_raw), to_screen(v2_raw),
            fill_opacity=0.3, fill_color="#FF00FF", stroke_width=1, color="#FF00FF"
        )
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        summary_text = Text("Cramer's Rule: A Ratio of Areas", font_size=24, color="#FFFFFF")
        # Issue 31: Use place_in_area for centering
        self.place_in_area(summary_text, 'A1', 'A6', scale_factor=1.0)
        
        self.play(Write(summary_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        self.play(Create(v1_vec), Create(v2_vec), Create(b_vec), Create(poly_a))
        self.play(Write(v1_label), Write(v2_label), Write(b_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        dim_text = Text("Works for n-dimensions", font_size=18, color="#00FF00")
        # Issue 32: Place at B1 to avoid overlap with b head
        self.place_at_grid(dim_text, 'B1', scale_factor=0.8)
        self.play(FadeIn(dim_text))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFFF00"))
        
        # Drone movement using Asset
        # Issue 22: Integrate [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/drone.svg]
        drone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/drone.svg").scale(0.3)
        drone.move_to(origin_pos)
        
        self.play(FadeIn(drone))
        
        # Path: 8/3 * v1 then 5/3 * v2
        p1_raw = (8/3) * v1_raw
        p2_raw = p1_raw + (5/3) * v2_raw 
        
        path_line1 = Line(origin_pos, to_screen(p1_raw), color="#00FF00", stroke_width=2, stroke_opacity=0.5)
        path_line2 = Line(to_screen(p1_raw), to_screen(p2_raw), color="#0000FF", stroke_width=2, stroke_opacity=0.5)
        
        self.play(
            drone.animate.move_to(to_screen(p1_raw)),
            Create(path_line1),
            run_time=1.5
        )
        self.play(
            drone.animate.move_to(to_screen(p2_raw)),
            Create(path_line2),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF00FF"))
        
        # Final cleanup: Fade out geometry, keep summary
        self.play(
            FadeOut(v1_vec), FadeOut(v2_vec), FadeOut(b_vec),
            FadeOut(v1_label), FadeOut(v2_label), FadeOut(b_label),
            FadeOut(poly_a), FadeOut(dim_text),
            FadeOut(drone), 
            FadeOut(path_line1), FadeOut(path_line2),
            # Transition summary text to final centered position
            summary_text.animate.scale(1.2).move_to(self.grid["C3"])
        )
        self.wait(2)
