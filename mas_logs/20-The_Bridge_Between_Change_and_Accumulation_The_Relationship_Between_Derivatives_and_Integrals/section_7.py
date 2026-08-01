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

class Section7Scene(TeachingScene):
    def construct(self):
        # Setup the layout with the specific title and lines
        lines = [
            'Differentiation and integration are two sides of one coin.',
            'They function as inverse operations, reversing each other.',
            'Together, they link momentary change to total accumulation.'
        ]
        self.setup_layout("Summary and Conclusion", lines)

        # === Animation for Lecture Line 1 ===
        # Step 1: Display the words 'Differentiation' and 'Integration'
        # Color: White (#FFFFFF)
        self.play(self.lecture[0].animate.set_color(WHITE), run_time=0.5)
        
        diff_text = Text("Differentiation", font_size=24, color=WHITE)
        int_text = Text("Integration", font_size=24, color=WHITE)
        
        # Place in areas to ensure centering and avoid point-based crowding
        self.place_in_area(diff_text, 'B1', 'B3', scale_factor=0.8)
        self.place_in_area(int_text, 'B4', 'B6', scale_factor=0.8)
        
        self.play(Write(diff_text), Write(int_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Step 2: Draw a circular orange arrow (#FC6255) connecting the two terms
        self.play(self.lecture[1].animate.set_color("#FC6255"), run_time=0.5)
        
        # Create a loop using two curved arrows
        arrow_top = CurvedArrow(
            start_point=diff_text.get_top() + UP * 0.1, 
            end_point=int_text.get_top() + UP * 0.1, 
            angle=-TAU/4, 
            color="#FC6255"
        )
        arrow_bottom = CurvedArrow(
            start_point=int_text.get_bottom() + DOWN * 0.1, 
            end_point=diff_text.get_bottom() + DOWN * 0.1, 
            angle=-TAU/4, 
            color="#FC6255"
        )
        
        self.play(Create(arrow_top), Create(arrow_bottom))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Step 3: Display the sequences s -> v -> a and a -> v -> s
        # Color: Blue (#58C4DD)
        self.play(self.lecture[2].animate.set_color("#58C4DD"), run_time=0.5)
        
        # Physics sequences using Text to bypass LaTeX dependency error
        seq1 = Text("s → v → a", color="#58C4DD", font_size=24)
        seq2 = Text("a → v → s", color="#58C4DD", font_size=24)
        
        # Place them in the lower half of the right side with adjusted scaling
        self.place_in_area(seq1, "D2", "D5", scale_factor=1.0)
        self.place_in_area(seq2, "E2", "E5", scale_factor=1.0)
        
        self.play(Write(seq1), Write(seq2))
        self.wait(2)
