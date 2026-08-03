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
        # Initial Setup
        title = "Linear Dependence: The Redundant Helper"
        lines = [
            "Linear dependence occurs when a vector adds no new directions.",
            "If a vector is already in the span, it's redundant.",
            "Three vectors on a flat floor are linearly dependent."
        ]
        self.setup_layout(title, lines)
        
        # Colors
        color_v = "#00FF00"
        color_w = "#0000FF"
        color_u = "#FF0000"
        
        # === Animation for Lecture Line 1 ===
        # Show vectors v (1,0) in #00FF00 and w (0,1) in #0000FF.
        self.lecture[0].set_color(YELLOW)
        
        plane = NumberPlane(
            x_range=[-2, 2, 1], y_range=[-2, 2, 1],
            x_length=5, y_length=5,
            background_line_style={"stroke_opacity": 0.3}
        )
        
        v_vec = Arrow(plane.c2p(0,0), plane.c2p(1,0), buff=0, color=color_v)
        w_vec = Arrow(plane.c2p(0,0), plane.c2p(0,1), buff=0, color=color_w)
        v_label = MathTex("\\vec{v}", color=color_v, font_size=24).next_to(v_vec.get_end(), DOWN, buff=0.1)
        w_label = MathTex("\\vec{w}", color=color_w, font_size=24).next_to(w_vec.get_end(), LEFT, buff=0.1)
        
        # Group components for positioning (Issue 27: larger area A1-F6)
        visual_group = VGroup(plane, v_vec, w_vec, v_label, w_label)
        self.place_in_area(visual_group, "A1", "F6", scale_factor=0.9)
        
        self.play(Create(plane))
        self.play(GrowArrow(v_vec), Write(v_label))
        self.play(GrowArrow(w_vec), Write(w_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight the 2D plane formed by the span of v and w [Asset: floor.svg].
        # Introduce vector u (1,1) in #FF0000 labeled 'North-East'.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Asset Integration (Issue 17)
        floor_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/floor.svg")
        floor_asset.set_opacity(0.2)
        floor_asset.set_color(WHITE)
        # Scale to match the plane area
        floor_asset.set_width(plane.x_length * 0.9)
        floor_asset.move_to(plane.get_center())
        
        u_vec = Arrow(plane.c2p(0,0), plane.c2p(1,1), buff=0, color=color_u)
        u_label = Text("North-East", color=color_u, font_size=20).next_to(u_vec.get_end(), UR, buff=0.1)
        
        self.play(FadeIn(floor_asset))
        self.play(GrowArrow(u_vec), Write(u_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show u perfectly aligning with the sum of v + w.
        # Flash u, v, and w to indicate 'Linear Dependence' and redundancy.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Visualize the sum v + w
        w_ghost = Arrow(plane.c2p(1,0), plane.c2p(1,1), buff=0, color=color_w, stroke_width=2).set_opacity(0.5)
        v_ghost = Arrow(plane.c2p(0,1), plane.c2p(1,1), buff=0, color=color_v, stroke_width=2).set_opacity(0.5)
        
        self.play(Create(w_ghost), Create(v_ghost))
        self.wait(0.5)
        
        # Flash to indicate alignment
        self.play(Flash(u_vec.get_end(), color=color_u, line_length=0.3))
        self.play(
            Indicate(u_vec, color=color_u), 
            Indicate(v_vec, color=color_v), 
            Indicate(w_vec, color=color_w),
            Indicate(floor_asset, color=YELLOW)
        )
        
        # Redundancy text
        redundant_text = Text("REDUNDANT", color=color_u, font_size=24, weight=BOLD).next_to(u_label, UP, buff=0.2)
        self.play(Write(redundant_text))
        
        self.wait(2)
        self.lecture[2].set_color(WHITE)
        self.wait(1)
