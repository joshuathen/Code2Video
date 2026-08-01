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
        # Setup the layout with title and 5 lecture lines
        self.setup_layout(
            "The Great Divide: Explicit vs. Implicit", 
            [
                "In explicit functions, y is already solved for x.",
                "It's like a puzzle where every piece fits perfectly.",
                "Implicit equations keep x and y tangled together.",
                "Here, x and y are inseparable partners.",
                "We can still find the slope for these curves."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Line: In explicit functions, y is already solved for x.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        eq1 = Text("y = x²", color=WHITE, font_size=32)
        self.place_in_area(eq1, "A1", "A3", scale_factor=1.0)
        
        label1 = Text("Explicit", color=YELLOW, font_size=24)
        # Fix Issue 31: Adjusted area to prevent clipping and scale factor
        self.place_in_area(label1, "A4", "A5", scale_factor=0.8)
        
        self.play(Write(eq1), Write(label1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line: It's like a puzzle where every piece fits perfectly.
        self.play(self.lecture[1].animate.set_color(BLUE))
        
        # Solve Puzzle Box
        box = Rectangle(height=0.8, width=1.5, color=BLUE)
        self.place_at_grid(box, "B4")
        box_text = Text("Solved", font_size=18, color=BLUE)
        box_text.next_to(box, DOWN, buff=0.1)
        
        puzzle_piece = Square(side_length=0.4, fill_opacity=1, color=YELLOW)
        y_text = Text("y", color=BLACK, font_size=24)
        piece_group = VGroup(puzzle_piece, y_text)
        
        # Fix Issue 33: Place closer to the box (B3 instead of B1) to ensure spatial connection
        self.place_at_grid(piece_group, "B3", scale_factor=0.6)

        self.play(Create(box), Write(box_text))
        self.play(piece_group.animate.move_to(self.grid["B4"]))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line: Implicit equations keep x and y tangled together.
        self.play(self.lecture[2].animate.set_color(PURPLE))
        
        eq2 = Text("x² + y² = 25", color="#00FF00", font_size=32)
        self.place_in_area(eq2, "C1", "C3", scale_factor=1.0)
        
        label2 = Text("Implicit", color=PURPLE, font_size=24)
        # Fix Issue 31: Adjusted area to prevent clipping and scale factor
        self.place_in_area(label2, "C4", "C5", scale_factor=0.8)
        
        self.play(Write(eq2), Write(label2))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Line: Here, x and y are inseparable partners.
        self.play(self.lecture[3].animate.set_color(WHITE))
        
        # Tangled x and y labels
        x_label = Text("x", color=WHITE, font_size=36)
        y_label = Text("y", color=WHITE, font_size=36)
        
        # Position them relative to D3 (center of action for this line)
        center_pos = self.grid["D3"]
        x_label.move_to(center_pos + LEFT*0.3)
        y_label.move_to(center_pos + RIGHT*0.3)
        tangled_group = VGroup(x_label, y_label)

        self.play(Create(tangled_group))
        self.play(Rotating(tangled_group, about_point=center_pos, radians=2*PI, run_time=2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Line: We can still find the slope for these curves.
        self.play(self.lecture[4].animate.set_color(TEAL))
        
        # Dashed line slope representing rate of change
        slope_line = DashedLine(
            start=LEFT*0.5, end=RIGHT*0.5, color=TEAL
        ).rotate(45*DEGREES)
        # Fix Issue 32: Repositioned to D2 with scale 1.0 to avoid overlap and buffer issues
        self.place_at_grid(slope_line, "D2", scale_factor=1.0)
        
        self.play(Create(slope_line))
        self.wait(2)
