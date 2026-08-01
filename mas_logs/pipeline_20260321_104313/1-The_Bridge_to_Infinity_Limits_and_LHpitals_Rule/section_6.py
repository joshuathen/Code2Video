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
        # Setup the layout with the title and the three lecture lines
        self.setup_layout("Summary and Synthesis", [
            "Limits bridge the gaps where functions are undefined.",
            "Rigorous proofs and growth rates solve these mysteries.",
            "You can now navigate the bridge to infinity confidently."
        ])

        # === Animation for Lecture Line 1 ===
        # Highlight Line 1
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        
        # Display two white line segments with a gap in the center.
        # Line(LEFT, RIGHT) has a default length of 2.0.
        # place_in_area(..., 'C1', 'C2') centers it at grid x=1.0, spanning 0.0 to 2.0.
        # place_in_area(..., 'C5', 'C6') centers it at grid x=5.0, spanning 4.0 to 6.0.
        # This leaves a gap from x=2.0 to x=4.0.
        seg_left = Line(LEFT, RIGHT, color="#FFFFFF", stroke_width=8)
        self.place_in_area(seg_left, "C1", "C2")
        
        seg_right = Line(LEFT, RIGHT, color="#FFFFFF", stroke_width=8)
        self.place_in_area(seg_right, "C5", "C6")
        
        self.play(Create(seg_left), Create(seg_right))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight Line 2 (Blue to match the microscope)
        self.play(self.lecture[1].animate.set_color("#3388FF"))
        
        # Create a blue microscope icon (#3388FF)
        microscope = VGroup(
            Line(LEFT, RIGHT, stroke_width=6).scale(0.3), # Base
            Line(LEFT, RIGHT, stroke_width=6).scale(0.4).rotate(60*DEGREES).shift(UP*0.2), # Body
            Circle(radius=0.1, stroke_width=6).shift(UP*0.4 + RIGHT*0.1) # Eyepiece
        ).set_color("#3388FF")
        
        # Create an orange wrench icon (#FFAA00)
        wrench = VGroup(
            Rectangle(width=0.1, height=0.5, stroke_width=6), # Handle
            Annulus(inner_radius=0.1, outer_radius=0.25, stroke_width=6).shift(UP*0.3) # Head
        ).set_color("#FFAA00")
        
        # Start icons at the bottom (Row F) and float them into the gap area (Row C)
        self.place_at_grid(microscope, "F3")
        self.place_at_grid(wrench, "F4")
        
        self.play(FadeIn(microscope), FadeIn(wrench))
        self.play(
            microscope.animate.move_to(self.grid["C3"]),
            wrench.animate.move_to(self.grid["C4"])
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight Line 3 (Green to match the fixed bridge)
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        
        # The gap fills with a solid green segment (#00FF00).
        # place_in_area(..., 'C3', 'C4') centers at x=3.0, spanning 2.0 to 4.0.
        # This perfectly connects the white segments at x=2.0 and x=4.0.
        bridge_fill = Line(LEFT, RIGHT, color="#00FF00", stroke_width=8)
        self.place_in_area(bridge_fill, "C3", "C4")
        
        # Completing the bridge: tools fade out as the gap is filled
        self.play(
            FadeOut(microscope),
            FadeOut(wrench),
            Create(bridge_fill)
        )
        self.wait(3)
