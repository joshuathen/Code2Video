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
        # Setup the layout with specific lecture lines
        lecture_lines = [
            "Factorials also count how many ways to arrange objects.",
            "To arrange zero items, there is only one way.",
            "We simply leave the shelf empty, creating one unique state."
        ]
        self.setup_layout("The Combinatorial Approach (The Empty Set)", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight Line 1
        self.lecture[0].set_color(WHITE)
        
        # Create boxes 1 and 2
        box1 = VGroup(Square(side_length=0.8, color=WHITE), Text("1", font_size=24, color=WHITE))
        box2 = VGroup(Square(side_length=0.8, color=WHITE), Text("2", font_size=24, color=WHITE))
        
        self.place_at_grid(box1, "B2")
        self.place_at_grid(box2, "B3")
        
        self.play(FadeIn(box1), FadeIn(box2))
        self.wait(1)
        
        # Animate swapping positions
        self.play(
            box1.animate.move_to(self.grid["B3"]),
            box2.animate.move_to(self.grid["B2"]),
            run_time=1.5
        )
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # Highlight Line 2
        self.play(
            self.lecture[0].animate.set_color(GREY),
            self.lecture[1].animate.set_color(WHITE)
        )
        
        # Clear boxes and show a shelf
        self.play(FadeOut(box1), FadeOut(box2))
        
        # Define shelf line (horizontal line)
        shelf = Line(self.grid["C1"], self.grid["C5"], color=WHITE)
        
        # Formula text - Replaced MathTex with Text due to missing LaTeX environment
        formula_q = Text("0! = ?", font_size=36, color=WHITE)
        self.place_at_grid(formula_q, "E3")
        
        self.play(Create(shelf), Write(formula_q))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight Line 3 in Gold
        self.play(
            self.lecture[1].animate.set_color(GREY),
            self.lecture[2].animate.set_color("#FFD700")
        )
        
        # Translucent square representing the 'empty set' state
        empty_state_box = Square(side_length=0.8, fill_opacity=0.3, stroke_opacity=0.5, color=WHITE)
        self.place_at_grid(empty_state_box, "B3") # Place on the shelf area
        
        # Final formula in Gold - Replaced MathTex with Text
        formula_final = Text("0! = 1", font_size=36, color="#FFD700")
        self.place_at_grid(formula_final, "E3")
        
        self.play(FadeIn(empty_state_box))
        self.play(Transform(formula_q, formula_final))
        self.wait(2)
