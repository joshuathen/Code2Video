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

class Section4Scene(TeachingScene):
    def construct(self):
        # Setup the scene layout
        self.setup_layout(
            "The Redundancy: Linear Dependence", 
            [
                "Some vectors don't add new directions.", 
                "If one vector is redundant, they're dependent.", 
                "Independent vectors each provide a unique path."
            ]
        )

        # Initialize visual elements
        # A simple plane representing the span of North and East movements
        plane = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            background_line_style={"stroke_opacity": 0.3},
            axis_config={"stroke_opacity": 0.5}
        )
        self.place_in_area(plane, 'B2', 'E5', scale_factor=0.8)

        # Vector v: Drone A (North) - Cyan
        v = Arrow(
            plane.c2p(0, 0), plane.c2p(0, 1.5), 
            buff=0, color="#00FFFF", stroke_width=6
        )
        v_label = Text("v: North", font_size=16, color="#00FFFF")
        v_label.next_to(v.get_end(), LEFT, buff=0.1)

        # Vector w: Drone B (East) - Yellow
        w = Arrow(
            plane.c2p(0, 0), plane.c2p(1.5, 0), 
            buff=0, color="#FFFF00", stroke_width=6
        )
        w_label = Text("w: East", font_size=16, color="#FFFF00")
        w_label.next_to(w.get_end(), DOWN, buff=0.1)

        # Vector u: Drone C (North-East) - Magenta
        u = Arrow(
            plane.c2p(0, 0), plane.c2p(1.5, 1.5), 
            buff=0, color="#FF00FF", stroke_width=6
        )
        u_label = Text("u: NE (Redundant)", font_size=16, color="#FF00FF")
        u_label.next_to(u.get_end(), UR, buff=0.1)

        # Ghost paths for visualization of combination
        v_ghost = Arrow(
            plane.c2p(1.5, 0), plane.c2p(1.5, 1.5), 
            buff=0, color="#00FFFF", stroke_opacity=0.4, stroke_width=4
        )
        w_ghost = Arrow(
            plane.c2p(0, 1.5), plane.c2p(1.5, 1.5), 
            buff=0, color="#FFFF00", stroke_opacity=0.4, stroke_width=4
        )

        # === Animation for Lecture Line 1 ===
        # Line: "Some vectors don't add new directions."
        self.play(self.lecture[0].animate.set_color("#00FFFF"), run_time=1)
        self.play(
            Create(plane),
            GrowArrow(v),
            FadeIn(v_label),
            GrowArrow(w),
            FadeIn(w_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line: "If one vector is redundant, they're dependent."
        self.play(self.lecture[1].animate.set_color("#FF00FF"), run_time=1)
        
        # Show that u is just a combination of v and w
        self.play(Create(v_ghost), Create(w_ghost), run_time=1.5)
        self.play(GrowArrow(u), FadeIn(u_label), run_time=1)
        self.play(Indicate(u, color="#FF00FF"), run_time=1.5)
        
        # Dim the ghosts to focus on the vectors
        self.play(v_ghost.animate.set_stroke(opacity=0.1), w_ghost.animate.set_stroke(opacity=0.1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line: "Independent vectors each provide a unique path."
        self.play(self.lecture[2].animate.set_color("#FFFF00"), run_time=1)
        
        # Remove the redundant vector to show independence of the remaining two
        self.play(
            FadeOut(u),
            FadeOut(u_label),
            FadeOut(v_ghost),
            FadeOut(w_ghost),
            run_time=1.5
        )
        
        # Highlight that v and w are unique paths
        self.play(
            v.animate.scale(1.1).set_stroke(width=8),
            w.animate.scale(1.1).set_stroke(width=8),
            run_time=1
        )
        self.play(
            v.animate.scale(1/1.1).set_stroke(width=6),
            w.animate.scale(1/1.1).set_stroke(width=6),
            run_time=1
        )
        self.wait(2)
