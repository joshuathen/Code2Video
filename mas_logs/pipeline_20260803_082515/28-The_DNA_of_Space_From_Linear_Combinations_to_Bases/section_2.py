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

class Section2Scene(TeachingScene):
    def construct(self):
        # Title and Lecture lines from storyboard
        title_text = "Linear Combinations: The Vector Recipe"
        lecture_lines = [
            "A linear combination scales vectors before adding them.",
            "It's a recipe for reaching any point in space.",
            "Our robot combines three steps east and two north.",
            "This sum reaches the treasure at coordinate three, two."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors from storyboard
        v_color = "#00FF00"  # Green
        w_color = "#0000FF"  # Blue
        res_color = "#FF00FF" # Magenta
        
        # === Animation for Lecture Line 1 ===
        # A linear combination scales vectors before adding them.
        self.lecture[0].set_color(YELLOW)
        
        # Create a 2D grid with the origin at center of its visual area.
        # Note: grid_plane is just visual background here.
        grid_plane = NumberPlane(
            x_range=[-1, 5, 1],
            y_range=[-1, 5, 1],
            x_length=6,
            y_length=6,
            background_line_style={"stroke_opacity": 0.2},
            axis_config={"stroke_opacity": 0.5}
        )
        self.place_in_area(grid_plane, 'A1', 'F6')
        
        # Base vectors: v (1,0) and w (0,1)
        # Origin is E2
        v_vec = Arrow(self.grid['E2'], self.grid['E3'], buff=0, color=v_color)
        w_vec = Arrow(self.grid['E2'], self.grid['D2'], buff=0, color=w_color)
        
        v_label = MathTex("\\vec{v}", color=v_color, font_size=24)
        self.place_at_grid(v_label, 'F3')
        
        # Fix for Issue 23: Move w_label to D3 (away from left edge)
        w_label = MathTex("\\vec{w}", color=w_color, font_size=24)
        self.place_at_grid(w_label, 'D3', scale_factor=0.8)
        
        self.play(Create(grid_plane))
        self.play(Create(v_vec), Write(v_label))
        self.play(Create(w_vec), Write(w_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # It's a recipe for reaching any point in space.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Visualizing "any point" with quick dots
        random_points = ['B3', 'C4', 'A5', 'D6', 'B2', 'F4']
        dots = VGroup(*[Dot(self.grid[p], color=YELLOW, radius=0.05) for p in random_points])
        self.play(Create(dots), run_time=1)
        self.play(FadeOut(dots))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Our robot combines three steps east and two north.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Extend v to 3v (E2 to E5)
        v3_vec = Arrow(self.grid['E2'], self.grid['E5'], buff=0, color=v_color)
        v3_label = MathTex("3\\vec{v}", color=v_color, font_size=24)
        self.place_at_grid(v3_label, 'F5')
        
        # Extend w to 2w (E2 to C2)
        w2_vec = Arrow(self.grid['E2'], self.grid['C2'], buff=0, color=w_color)
        # Fix for Issue 24: Move w2_label to C3 (away from boundary)
        w2_label = MathTex("2\\vec{w}", color=w_color, font_size=24)
        self.place_at_grid(w2_label, 'C3', scale_factor=0.8)
        
        self.play(
            Transform(v_vec, v3_vec),
            Transform(v_label, v3_label),
            Transform(w_vec, w2_vec),
            Transform(w_label, w2_label)
        )
        self.wait(1)
        
        # Tip-to-tail: Move 2w to start at E5 and end at C5
        w_tt_vec = Arrow(self.grid['E5'], self.grid['C5'], buff=0, color=w_color)
        w_tt_label = MathTex("2\\vec{w}", color=w_color, font_size=24)
        self.place_at_grid(w_tt_label, 'C6')
        
        self.play(
            Transform(w_vec, w_tt_vec),
            Transform(w_label, w_tt_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # This sum reaches the treasure at coordinate three, two.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Draw the resultant vector (3,2) in #FF00FF
        res_vec = Arrow(self.grid['E2'], self.grid['C5'], buff=0, color=res_color)
        res_label = Text("Linear Combination", color=res_color, font_size=18)
        # Fix for Issue 22: Place res_label in area B4-B6 to avoid cramping
        self.place_in_area(res_label, 'B4', 'B6', scale_factor=0.6)
        
        # Treasure at C5
        treasure = Star(color=YELLOW, fill_opacity=1, stroke_width=1).scale(0.15)
        self.place_at_grid(treasure, 'C5')
        
        self.play(Create(res_vec), Write(res_label))
        self.play(FadeIn(treasure))
        self.wait(3)
