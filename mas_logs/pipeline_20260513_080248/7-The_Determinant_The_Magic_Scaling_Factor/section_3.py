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

class Section3Scene(TeachingScene):
    def construct(self):
        # Define the lecture script
        lecture_lines = [
            'This unit square represents a starting area of one.', 
            'The transformation reshapes the square into a parallelogram.', 
            'Observe the area has now scaled to six.', 
            'This scaling factor applies to every shape in space.', 
            'The determinant represents this magic area scaling factor.'
        ]
        
        # Initialize layout
        self.setup_layout("Defining the Determinant Geometrically", lecture_lines)
        
        # Matrix to be used for transformations: [[3, 1], [0, 2]]
        trans_matrix = [[3, 1, 0], [0, 2, 0], [0, 0, 1]]
        yellow_hex = "#FFFF00"
        blue_hex = BLUE_B
        
        square_asset = "/mmfs1/data/home/jthen/Code2Video/assets/icon/square.svg"
        circle_asset = "/mmfs1/data/home/jthen/Code2Video/assets/icon/circle.svg"
        
        # === Animation for Lecture Line 1 ===
        # Highlight a unit square area and label it 'Area = 1'.
        self.lecture[0].set_color(blue_hex)
        
        unit_square = SVGMobject(square_asset)
        unit_square.set_fill(blue_hex, opacity=0.4)
        unit_square.set_stroke(blue_hex, width=2)
        self.place_at_grid(unit_square, "D3")
        
        label_1 = Text("Area = 1", font_size=24, color=WHITE)
        self.place_at_grid(label_1, "E3")
        
        self.play(
            Create(unit_square),
            Write(label_1),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transform the square into a parallelogram using matrix [[3, 1], [0, 2]].
        self.lecture[1].set_color(yellow_hex)
        
        parallelogram = SVGMobject(square_asset)
        parallelogram.set_fill(yellow_hex, opacity=0.4)
        parallelogram.set_stroke(yellow_hex, width=2)
        parallelogram.apply_matrix(trans_matrix)
        # Issue 38: Use scale_factor=0.7 to avoid obstruction of labels
        self.place_at_grid(parallelogram, "D3", scale_factor=0.7)
        
        self.play(
            ReplacementTransform(unit_square, parallelogram),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Animate the label 'Area = 1' changing into 'Area = 6' (#FFFF00).
        self.lecture[2].set_color(yellow_hex)
        
        label_6 = Text("Area = 6", font_size=24, color=yellow_hex)
        self.place_at_grid(label_6, "E3")
        
        self.play(
            ReplacementTransform(label_1, label_6),
            run_time=1.5
        )
        self.wait(1.5)

        # === Animation for Lecture Line 4 ===
        # Draw a circle; transform it into an ellipse using the same matrix.
        self.lecture[3].set_color(blue_hex)
        
        # Prepare space for the circle
        self.play(FadeOut(parallelogram), FadeOut(label_6), run_time=0.8)
        
        circle = SVGMobject(circle_asset)
        circle.set_fill(blue_hex, opacity=0.4)
        circle.set_stroke(blue_hex, width=2)
        self.place_at_grid(circle, "D3")
        
        self.play(Create(circle))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Display text 'Determinant = 6' at the top of the screen in #FFFF00.
        self.lecture[4].set_color(yellow_hex)
        
        ellipse = SVGMobject(circle_asset)
        ellipse.set_fill(yellow_hex, opacity=0.4)
        ellipse.set_stroke(yellow_hex, width=2)
        ellipse.apply_matrix(trans_matrix)
        # Issue 40: Set ellipse scale to 0.7 for consistency
        self.place_at_grid(ellipse, "D3", scale_factor=0.7)
        
        det_label = Text("Determinant = 6", font_size=32, color=yellow_hex)
        # Issue 39: Position label at C2-C5 for correct proximity
        self.place_in_area(det_label, "C2", "C5", scale_factor=0.8)
        
        self.play(
            ReplacementTransform(circle, ellipse),
            Write(det_label),
            run_time=2
        )
        self.wait(2)
