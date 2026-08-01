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

class Section5Scene(TeachingScene):
    def construct(self):
        # Fetching lecture lines and title from storyboard
        title = "Updating the Belief: The Posterior"
        lecture_lines = [
            "The new probability is the ratio of these areas.",
            "We compare the Phoenix rectangle to the total evidence.",
            "This process 'normalizes' our belief to the new evidence.",
            "Geometrically, it is the Phoenix portion of the evidence.",
            "Our updated belief is called the Posterior Probability."
        ]
        self.setup_layout(title, lecture_lines)
        
        # Colors
        BLUE_COLOR = "#3498db"
        YELLOW_COLOR = "#f1c40f"
        WHITE_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # "The new probability is the ratio of these areas."
        self.lecture[0].set_color(BLUE_COLOR)
        
        # Representing the components from the previous section
        phoenix_rect = Rectangle(width=2.2, height=1.6, fill_color=BLUE_COLOR, fill_opacity=0.8, stroke_color=BLUE_COLOR)
        fluke_rect = Rectangle(width=2.2, height=0.8, fill_color=YELLOW_COLOR, fill_opacity=0.8, stroke_color=YELLOW_COLOR)
        
        # Labels for clarity
        phoenix_label = Text("Phoenix + Evidence", font_size=16, color=BLUE_COLOR)
        fluke_label = Text("Fluke + Evidence", font_size=16, color=YELLOW_COLOR)
        
        # Initial positions
        self.place_at_grid(phoenix_rect, "B3")
        self.place_at_grid(fluke_rect, "D3")
        phoenix_label.next_to(phoenix_rect, RIGHT, buff=0.2)
        fluke_label.next_to(fluke_rect, RIGHT, buff=0.2)

        self.play(
            FadeIn(phoenix_rect), 
            FadeIn(fluke_rect), 
            FadeIn(phoenix_label), 
            FadeIn(fluke_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "We compare the Phoenix rectangle to the total evidence."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW_COLOR)
        
        # Storyboard: "Move the blue rectangle to the top and the yellow one below it."
        stack = VGroup(phoenix_rect, fluke_rect).arrange(DOWN, buff=0)
        
        self.play(
            stack.animate.move_to(self.grid["C3"]),
            FadeOut(phoenix_label),
            FadeOut(fluke_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "This process 'normalizes' our belief to the new evidence."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(WHITE_COLOR)
        
        # Storyboard: "Draw a white bracket (#FFFFFF) encompassing both to represent 'Total Evidence'."
        bracket = Brace(stack, direction=RIGHT, color=WHITE_COLOR)
        bracket_label = Text("Total Evidence", font_size=18, color=WHITE_COLOR)
        bracket_label.next_to(bracket, RIGHT, buff=0.1)
        
        self.play(Create(bracket), FadeIn(bracket_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Geometrically, it is the Phoenix portion of the evidence."
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(BLUE_COLOR)
        
        # Storyboard: "Scale the blue rectangle to fill its proportion of the total stack."
        # We simulate this by showing the ratio in a formula.
        
        # Formula showing the ratio - Fix Issue #32: Use place_in_area for horizontal balance
        formula = MathTex(
            r"P(\text{Phoenix} | \text{Evidence}) = \frac{\text{Area(Blue)}}{\text{Area(Total)}}",
            font_size=24, color=WHITE
        )
        # Applying requested fix: self.place_in_area(formula, 'E2', 'E4', scale_factor=0.7)
        self.place_in_area(formula, 'E2', 'E4', scale_factor=0.7)

        self.play(
            Write(formula),
            run_time=2
        )
        
        # Highlight the Phoenix portion as the outcome
        self.play(Indicate(phoenix_rect, color=BLUE_COLOR))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "Our updated belief is called the Posterior Probability."
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(WHITE_COLOR)
        
        # Label the concept - Fix Issue #33: Use place_in_area for better centering
        posterior_label = Text("Posterior Probability", font_size=24, color=BLUE_COLOR)
        # Applying requested fix: self.place_in_area(posterior_label, 'F2', 'F4', scale_factor=0.8)
        self.place_in_area(posterior_label, 'F2', 'F4', scale_factor=0.8)
        
        self.play(Write(posterior_label))
        self.wait(3)
