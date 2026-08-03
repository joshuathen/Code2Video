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
        # Data from storyboard
        title_text = "The Loss Function: Measuring the 'Ouch'"
        lines = [
            "Loss measures the distance between guess and reality.",
            "High loss indicates a very inaccurate prediction.",
            "We want to minimize this error value."
        ]
        self.setup_layout(title_text, lines)
        
        # Colors
        pred_color = YELLOW
        truth_color = GREEN
        error_color = "#FF0000"
        loss_color = "#FF5555"
        
        # === Animation for Lecture Line 1 ===
        # Highlight current lecture line
        self.play(self.lecture[0].animate.set_color(BLUE))
        
        # Create Number Line
        num_line = NumberLine(
            x_range=[0, 1.2, 0.2],
            length=5,
            include_numbers=True,
            font_size=18,
            color=WHITE
        )
        self.place_in_area(num_line, "C1", "C6")
        
        # Prediction (Guess) and Truth points
        pred_dot = Dot(num_line.n2p(0.7), color=pred_color)
        truth_dot = Dot(num_line.n2p(1.0), color=truth_color)
        
        pred_label = Text("Guess: 0.7", font_size=16, color=pred_color)
        truth_label = Text("Truth: 1.0", font_size=16, color=truth_color)
        
        # Grid placement for labels
        # Fix 32: pred_label B4-B5 to align better with 0.7
        self.place_in_area(pred_label, 'B4', 'B5', scale_factor=0.8)
        self.place_at_grid(truth_label, "B6")
        
        self.play(Create(num_line))
        self.play(
            FadeIn(pred_dot), 
            FadeIn(pred_label),
            FadeIn(truth_dot), 
            FadeIn(truth_label)
        )
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # Highlight current lecture line
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(error_color)
        )
        
        # Visualize the gap/distance
        gap_line = Line(
            num_line.n2p(0.7), 
            num_line.n2p(1.0), 
            color=error_color, 
            stroke_width=10
        )
        error_text = Text("Error: 0.3", font_size=20, color=error_color)
        # Fix 33: error_text D4-D5 to center under the segment
        self.place_in_area(error_text, 'D4', 'D5', scale_factor=0.8)
        
        self.play(Create(gap_line))
        self.play(Write(error_text))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight current lecture line
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(loss_color)
        )
        
        # Visualize Loss using Asset (Issue 23)
        # Load the asset: [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/box.svg]
        loss_square = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/box.svg")
        
        # Visual size corresponds to the gap line's length
        visual_side = gap_line.get_length()
        loss_square.scale_to_fit_height(visual_side)
        loss_square.set_color(loss_color)
        loss_square.set_fill(loss_color, opacity=0.4)
        
        self.place_in_area(loss_square, "E3", "F4")
        
        loss_label = Text("Loss = 0.3^2 = 0.09", font_size=20, color=loss_color)
        # Fix 31: loss_label E5-F6 to avoid overlap with loss_square
        self.place_in_area(loss_label, 'E5', 'F6', scale_factor=0.7)
        
        self.play(FadeIn(loss_square))
        self.play(Write(loss_label))
        self.wait(2)
