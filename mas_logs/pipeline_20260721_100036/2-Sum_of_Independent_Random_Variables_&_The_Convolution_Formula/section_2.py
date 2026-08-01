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
        # Data from shared state
        title = "Prerequisite: The Discrete Logic"
        lecture_lines = [
            "For independent variables, the joint probability is their product.",
            "To find the sum's probability, sum all valid pairs.",
            "On a grid, these pairs form a diagonal line."
        ]
        
        self.setup_layout(title, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Show grid with axes labeled X and Y in #FFFFFF.
        # Joint probability is product -> highlight the whole grid idea.
        self.play(self.lecture[0].animate.set_color(WHITE)) # Active line color matching
        
        grid_dots = VGroup()
        for r in ["A", "B", "C", "D", "E", "F"]:
            for c in ["1", "2", "3", "4", "5", "6"]:
                dot = Dot(color=GRAY, radius=0.06)
                self.place_at_grid(dot, f"{r}{c}")
                grid_dots.add(dot)
                
        # Label axes X and Y
        x_axis_label = Text("X (Die 1)", font_size=20, color=WHITE)
        y_axis_label = Text("Y (Die 2)", font_size=20, color=WHITE).rotate(90 * DEGREES)
        
        # Fixed positioning as per Issue 47/33
        self.place_in_area(x_axis_label, 'F1', 'F6', scale_factor=0.6)
        x_axis_label.shift(DOWN * 0.7) # Small relative shift allowed for axis labels to avoid data overlap
        
        # Fixed positioning as per Issue 47/32
        self.place_in_area(y_axis_label, 'A1', 'F1', scale_factor=0.6)
        y_axis_label.shift(LEFT * 0.7) # Small relative shift allowed for axis labels to avoid data overlap
        
        self.play(
            Create(grid_dots),
            Write(x_axis_label),
            Write(y_axis_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight points (1,3), (2,2), (3,1) in #FFFF00 on the grid.
        self.play(
            self.lecture[1].animate.set_color("#FFFF00") # Matching color for points
        )
        
        # (1,3) -> D1, (2,2) -> E2, (3,1) -> F3
        target_cells = ["D1", "E2", "F3"]
        highlights = VGroup()
        coord_labels = VGroup()
        
        labels_info = {
            "D1": "(1,3)",
            "E2": "(2,2)",
            "F3": "(3,1)"
        }
        
        for pos in target_cells:
            h_dot = Dot(color="#FFFF00", radius=0.12)
            self.place_at_grid(h_dot, pos)
            highlights.add(h_dot)
            
            # Label
            label = Text(labels_info[pos], font_size=16, color="#FFFF00")
            # Apply fix from Issue 47/34: reduce scale and increase padding
            self.place_at_grid(label, pos, scale_factor=0.5)
            label.next_to(h_dot, UR, buff=0.15) 
            coord_labels.add(label)
            
        self.play(
            LaggedStart(*[Flash(h, color="#FFFF00") for h in highlights], lag_ratio=0.3),
            FadeIn(highlights),
            Write(coord_labels),
            run_time=2
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Draw a diagonal line through these points in #FF00FF.
        self.play(
            self.lecture[2].animate.set_color("#FF00FF") # Matching color for diagonal line
        )
        
        diag_line = Line(
            start=self.grid["D1"],
            end=self.grid["F3"],
            color="#FF00FF",
            stroke_width=6
        )
        diag_line.scale(1.5) # Extend slightly to emphasize the linear constraint x+y=4
        
        self.play(Create(diag_line), run_time=1.5)
        self.wait(3)
        
        # Final cleanup for color consistency
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(WHITE)
        )
        self.wait(1)
