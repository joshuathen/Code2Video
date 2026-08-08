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
        # Fetching title and lines from storyboard
        title = "The Big Idea: What is Convolution?"
        lines = [
            "Convolution mixes two signals to create a new one.",
            "One function's shape modifies the other's properties.",
            "Imagine a sliding window transforming data as it moves."
        ]
        self.setup_layout(title, lines)
        
        # === Animation for Lecture Line 1 ===
        # Create a grid of grey points #808080 representing 'Dusty Floor'.
        self.play(self.lecture[0].animate.set_color("#00FFFF")) # Highlight current line
        
        floor_dots = VGroup()
        dot_map = {}
        # Use a 4x4 grid subset for the floor area (B2 to E5)
        for row_char in ["B", "C", "D", "E"]:
            for col_char in ["2", "3", "4", "5"]:
                dot = Dot(color="#808080", radius=0.1)
                self.place_at_grid(dot, f"{row_char}{col_char}")
                floor_dots.add(dot)
                dot_map[f"{row_char}{col_char}"] = dot
        
        self.play(FadeIn(floor_dots))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Fade in the robot asset labeled 'Cleaning Robot (Kernel)' atop a blue square #0000FF.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#0000FF")
        )
        
        robot_square = Square(side_length=0.9, color="#0000FF", fill_opacity=0.3)
        self.place_at_grid(robot_square, "B2")
        
        robot_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        self.place_at_grid(robot_asset, "B2", scale_factor=0.6)
        
        robot_label = Text("Cleaning Robot", font_size=16, color="#0000FF")
        # VideoCritic fix: Use place_in_area for multi-word label
        self.place_in_area(robot_label, "A1", "A3", scale_factor=0.8)
        
        self.play(FadeIn(robot_square), FadeIn(robot_asset), FadeIn(robot_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Animate the robot asset sliding across the grey grid.
        # Change passed grey points to white #FFFFFF ('Clean Floor').
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFD700")
        )
        
        # Path sequence covering the floor dots
        path = [
            "B2", "B3", "B4", "B5", 
            "C5", "C4", "C3", "C2", 
            "D2", "D3", "D4", "D5", 
            "E5", "E4", "E3", "E2"
        ]
        
        for pos in path:
            row_char = pos[0]
            col_char = pos[1]
            
            # Determine label position (staying relative to the square)
            # Label always 1 unit above the current row
            label_row = chr(ord(row_char) - 1)
            # To maintain centering consistency with 'A1-A3' area, 
            # we move it to the grid point corresponding to the center of a 3-col span
            # For B2 center, label area is A1-A3, center is A2.
            # So label_pos is {label_row}{col_char}
            label_pos = f"{label_row}{col_char}"
            
            self.play(
                robot_square.animate.move_to(self.grid[pos]),
                robot_asset.animate.move_to(self.grid[pos]),
                robot_label.animate.move_to(self.grid[label_pos]),
                dot_map[pos].animate.set_color("#FFFFFF"),
                run_time=0.25
            )
        
        self.wait(0.5)
        
        # Display text 'Convolution = Input Signal ⊗ Filter Kernel' in gold #FFD700.
        formula = MathTex(
            r"\text{Convolution} = \text{Input Signal} \otimes \text{Filter Kernel}", 
            color="#FFD700"
        )
        # VideoCritic fix: use scale_factor=0.6 and place_in_area F1-F6
        self.place_in_area(formula, "F1", "F6", scale_factor=0.6)
        
        self.play(Write(formula))
        self.wait(2)
        
        # Final cleanup
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(2)
