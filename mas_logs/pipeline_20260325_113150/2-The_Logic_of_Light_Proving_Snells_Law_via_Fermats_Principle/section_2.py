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
        # 1. Initialize layout with the required title and lecture lines
        self.setup_layout("Prerequisite: Light Speed and Refractive Index", [
            "Light travels at different speeds depending on the medium.",
            "In glass, light moves slower than in a vacuum.",
            "We define velocity using the refractive index formula."
        ])

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1
        self.lecture[0].set_color(WHITE)
        
        # Define the medium (glass) area: D1 to F6
        glass_block = Rectangle(width=5.5, height=2.5, fill_color=BLUE_E, fill_opacity=0.3, stroke_width=0)
        self.place_in_area(glass_block, "D1", "F6")
        
        # Labels for clarity
        # Resolved Issue 36: Use area for better centering
        vac_label = Text("Vacuum (c)", font_size=20, color=WHITE)
        self.place_in_area(vac_label, "B1", "B3", scale_factor=0.8)
        
        # Resolved Issue 35: Scale down to avoid edge crowding
        glass_label = Text("Glass", font_size=20, color=WHITE)
        self.place_at_grid(glass_label, "D1", scale_factor=0.7)
        
        self.add(glass_block, vac_label, glass_label)

        # Ray path coordinates
        p_start = self.grid["A3"]
        p_interface = (self.grid["C3"] + self.grid["D3"]) / 2
        p_end = self.grid["F3"]
        
        # Ray segments: White in vacuum, Cyan in glass
        ray_vac = Line(p_start, p_interface, color="#FFFFFF", stroke_width=5)
        ray_glass = Line(p_interface, p_end, color="#00FFFF", stroke_width=5)
        
        # Animate the light ray entering the medium
        self.play(Create(ray_vac), run_time=1.0, rate_func=linear)
        self.play(Create(ray_glass), run_time=2.0, rate_func=linear) 
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Transition highlight to line 2 (dim previous)
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_color(WHITE)
        
        # Refractive index symbol inside the medium area
        n_label = Text("n > 1", color=WHITE, font_size=24)
        self.place_at_grid(n_label, "E2")
        
        self.play(Write(n_label))
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Transition highlight to line 3 (dim previous)
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color("#FFFF00") # Highlight in yellow
        
        # Define velocity formula
        v_formula = Text("v = c / n", color="#FFFF00", font_size=24)
        # Resolved Issue 34: Position in area for optimal centering
        self.place_in_area(v_formula, "E4", "F6", scale_factor=0.9)
        
        self.play(Write(v_formula))
        self.wait(2)
