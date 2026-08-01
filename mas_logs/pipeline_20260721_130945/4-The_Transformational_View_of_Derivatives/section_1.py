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

class Section1Scene(TeachingScene):
    def construct(self):
        # Data from shared state
        title = "The Shift: From Slope to Stretching"
        lecture_lines = [
            "Traditionally, we see derivatives as slopes on graphs.",
            "Let's view them as dynamic transformations instead.",
            "Functions map an input space to an output space."
        ]
        
        # Setup layout
        self.setup_layout(title, lecture_lines)
        
        # Colors for matching lecture lines and assets
        c_input = "#00FF00" 
        c_output = "#00BFFF"
        c_text = "#FFFFFF"
        
        # === Animation for Lecture Line 1 ===
        # Traditionally, we see derivatives as slopes on graphs.
        self.play(self.lecture[0].animate.set_color(c_text))
        
        # Asset: slope visualization [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/slope.svg]
        # Resolve Issue 26: Load and place the referenced SVG asset
        slope_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/slope.svg")
        slope_asset.set_color(c_text)
        self.place_in_area(slope_asset, "B2", "E6", scale_factor=1.2)
        
        self.play(FadeIn(slope_asset))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Let's view them as dynamic transformations instead.
        self.play(self.lecture[1].animate.set_color(c_input))
        
        # Fade out slope visualization
        self.play(FadeOut(slope_asset))
        
        # Display parallel lines "Input" (#00FF00) and "Output" (#00BFFF)
        # Resolve Issue 27: Position input_line in C3:C6 and input_label at C2
        input_line = Line(LEFT*1.5, RIGHT*1.5, color=c_input, stroke_width=4)
        self.place_in_area(input_line, 'C3', 'C6')
        
        input_label = Text("Input", font_size=18, color=c_input)
        self.place_at_grid(input_label, 'C2', scale_factor=0.8)
        
        # Resolve Issue 28: Position output_line in E3:E6 and output_label at E2
        output_line = Line(LEFT*1.5, RIGHT*1.5, color=c_output, stroke_width=4)
        self.place_in_area(output_line, 'E3', 'E6')
        
        output_label = Text("Output", font_size=18, color=c_output)
        self.place_at_grid(output_label, 'E2', scale_factor=0.8)
        
        self.play(
            Create(input_line),
            Create(output_line),
            FadeIn(input_label),
            FadeIn(output_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Functions map an input space to an output space.
        self.play(self.lecture[2].animate.set_color(c_output))
        
        # Draw mapping arrows from Input points to Output points
        arrows = VGroup()
        # Using grid column markers as anchor points for mapping arrows
        for col in ["3", "4", "5", "6"]:
            # Anchor positions derived from the same grid used for lines
            start_p = self.grid[f"C{col}"]
            end_p = self.grid[f"E{col}"]
            # Buffer slightly from the lines themselves
            arrow = Arrow(
                start_p + DOWN*0.1, 
                end_p + UP*0.1, 
                color=WHITE, 
                buff=0, 
                stroke_width=2, 
                max_tip_length_to_length_ratio=0.15
            )
            arrows.add(arrow)
            
        # Resolve Issue 29: Position scaling_factor label in D3:D6
        scaling_label = Text("Scaling Factor", font_size=20, color=c_text)
        self.place_in_area(scaling_label, 'D3', 'D6', scale_factor=0.8)
        
        self.play(
            Create(arrows),
            Write(scaling_label),
            run_time=2
        )
        self.wait(2)
