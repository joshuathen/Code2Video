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
        lecture_lines = [
            "The derivative measures the local scaling ratio.",
            "Values above one stretch space; below one shrink it.",
            "Negative derivatives flip the direction of the output."
        ]
        self.setup_layout("Defining the Derivative as a Scaling Factor", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # "The derivative measures the local scaling ratio."
        self.lecture[0].set_color(YELLOW)
        formula = MathTex(r"f'(x) = \frac{df}{dx}", color=WHITE)
        # Resolved Issue 33: Fix positioning and scaling
        self.place_in_area(formula, "A3", "B5", scale_factor=1.1)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Values above one stretch space; below one shrink it."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Resolved Issue 22: Integrate SVG asset
        gauge_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/gauge.svg")
        gauge_asset.set_color("#00FF00")
        
        # Resolved Issue 34: Fix gauge positioning
        self.place_in_area(gauge_asset, "C3", "D5", scale_factor=1.0)
        
        asset_center = gauge_asset.get_center()
        
        shrink_label = Text("Shrink (<1)", font_size=18, color="#00FF00")
        stretch_label = Text("Stretch (>1)", font_size=18, color="#00FF00")
        
        # Place labels near the asset
        shrink_label.next_to(gauge_asset, LEFT, buff=0.2)
        stretch_label.next_to(gauge_asset, RIGHT, buff=0.2)
        
        # Needle to show scaling
        needle = Arrow(asset_center, asset_center + LEFT * 0.8, color=WHITE, buff=0)
        
        self.play(
            FadeIn(gauge_asset), 
            FadeIn(shrink_label), 
            FadeIn(stretch_label), 
            Create(needle)
        )
        
        # Animate needle: Rotate to show different zones
        self.play(Rotate(needle, angle=-PI/4, about_point=asset_center), run_time=1)
        self.wait(0.5)
        self.play(Rotate(needle, angle=-PI/2, about_point=asset_center), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Negative derivatives flip the direction of the output."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Clear middle for the lines in bottom area
        self.play(
            FadeOut(gauge_asset), 
            FadeOut(shrink_label), 
            FadeOut(stretch_label), 
            FadeOut(needle)
        )
        
        input_line = Line(LEFT, RIGHT, color=WHITE).scale(2.2)
        output_line = Line(LEFT, RIGHT, color=WHITE).scale(2.2)
        
        self.place_in_area(input_line, "E1", "E6", scale_factor=1.0)
        self.place_in_area(output_line, "F1", "F6", scale_factor=1.0)
        
        # Labels for the lines
        input_label = Text("Input", font_size=16, color=WHITE).next_to(input_line, LEFT, buff=0.3)
        output_label = Text("Output", font_size=16, color=WHITE).next_to(output_line, LEFT, buff=0.3)
        
        # Flipped mapping arrows
        p1_start = input_line.get_left() + RIGHT * 1.0
        p2_start = input_line.get_right() - RIGHT * 1.0
        
        # Cross them
        p1_end = output_line.get_right() - RIGHT * 1.0
        p2_end = output_line.get_left() + RIGHT * 1.0
        
        arrow1 = Arrow(p1_start, p1_end, buff=0, color="#FF00FF")
        arrow2 = Arrow(p2_start, p2_end, buff=0, color="#FF00FF")
        
        flipped_text = Text("Flipped", font_size=22, color="#FF00FF")
        # Resolved Issue 35: Fix 'Flipped' text scaling to avoid overlap
        self.place_in_area(flipped_text, "E3", "F4", scale_factor=0.7)
        
        self.play(
            Create(input_line), 
            Create(output_line), 
            FadeIn(input_label), 
            FadeIn(output_label)
        )
        self.play(
            GrowArrow(arrow1), 
            GrowArrow(arrow2), 
            FadeIn(flipped_text)
        )
        self.wait(2)
