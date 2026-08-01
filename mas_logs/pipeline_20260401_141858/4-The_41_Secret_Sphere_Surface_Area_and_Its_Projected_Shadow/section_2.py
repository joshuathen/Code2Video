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
        # Setup layout
        lecture_lines = [
            "A circle's area depends on its radius.",
            "The formula for this area is π r squared.",
            "This circle captures light energy hitting the sphere."
        ]
        self.setup_layout("Prerequisite Knowledge: The Circle Base", lecture_lines)

        # Colors
        BLUE_DISC = "#0000FF"
        CYAN_RAD = "#00FFFF"
        YELLOW_FILL = "#FFFF00"
        WHITE_TEXT = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Display a flat blue disc with a radius line labeled 'r'
        # Incorporate Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/sphere.svg
        self.lecture[0].set_color(BLUE_DISC)
        
        circle = Circle(radius=1.5, color=BLUE_DISC, fill_opacity=0.3, stroke_width=4)
        
        # Load asset and position within the circle
        sphere_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/sphere.svg")
        sphere_icon.set_color(BLUE_DISC)
        sphere_icon.set_height(circle.get_height() * 0.7)
        sphere_icon.move_to(circle.get_center())
        
        radius_line = Line(circle.get_center(), circle.get_right(), color=CYAN_RAD)
        
        # Using Text for label to avoid LaTeX dependency
        r_label = Text("r", color=CYAN_RAD, font_size=36, font="Serif", slant=ITALIC)
        r_label.next_to(radius_line, UP, buff=0.1)
        
        circle_group = VGroup(circle, sphere_icon, radius_line, r_label)
        # ISSUE 35 FIX: Use A1-D6 area and 0.9 scale factor
        self.place_in_area(circle_group, "A1", "D6", scale_factor=0.9)
        
        self.play(Create(circle), FadeIn(sphere_icon), run_time=1)
        self.play(Create(radius_line), Write(r_label), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Fill the disc with a yellow color to visualize the area calculation
        self.lecture[1].set_color(YELLOW_FILL)
        
        yellow_fill = Circle(radius=1.5, color=YELLOW_FILL, fill_opacity=0.8, stroke_width=0)
        yellow_fill.move_to(circle.get_center())
        # Scale manually to match the scaling applied to circle_group in place_in_area (0.9)
        yellow_fill.scale(0.9) 
        
        self.play(FadeIn(yellow_fill), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Write the formula 'Area = πr²' in white and highlight it with a white box
        self.lecture[2].set_color(WHITE_TEXT)
        
        # Using Text for formula to avoid LaTeX dependency
        formula = Text("Area = πr²", color=WHITE_TEXT, font_size=40, font="Serif")
        box = SurroundingRectangle(formula, color=WHITE_TEXT, buff=0.2)
        formula_group = VGroup(box, formula)
        
        # ISSUE 34 & 36 FIX: Use scale factor 0.75 in area E2-F5
        self.place_in_area(formula_group, "E2", "F5", scale_factor=0.75)
        
        self.play(Write(formula), Create(box), run_time=1.5)
        self.wait(2)
