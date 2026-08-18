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
        # Data from storyboard
        title_text = "The Pizza Slicing Challenge"
        lecture_lines = [
            "Leo the Chef wants to maximize pizza slices.",
            "Place points on the circle edge and connect them.",
            "How many regions can we create with 'n' points?"
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        PIZZA_COLOR = WHITE
        POINT_COLOR = "#FFFF00"  # Yellow
        CHORD_COLOR = "#00FFFF"  # Cyan

        # === Animation for Lecture Line 1 ===
        # "Leo the Chef wants to maximize pizza slices."
        self.play(self.lecture[0].animate.set_color(PIZZA_COLOR))
        
        pizza = Circle(radius=2.12, color=PIZZA_COLOR)
        self.place_in_area(pizza, "B2", "E5")
        
        self.play(Create(pizza))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Place points on the circle edge and connect them."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(POINT_COLOR)
        )
        
        # Point 1 at B2
        p1 = Dot(color=POINT_COLOR)
        self.place_at_grid(p1, 'B2')
        
        # Initial region label '1' at the center of the pizza
        initial_label = Text("1", font_size=36, color=WHITE)
        self.place_in_area(initial_label, 'C3', 'D4', scale_factor=0.8) # Resolves Issue 22
        
        self.play(FadeIn(p1))
        self.play(Write(initial_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "How many regions can we create with 'n' points?"
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(CHORD_COLOR)
        )
        
        # Add second point at E5
        p2 = Dot(color=POINT_COLOR)
        self.place_at_grid(p2, 'E5')
        
        # Chord between p1 and p2
        chord = Line(self.grid['B2'], self.grid['E5'], color=CHORD_COLOR)
        
        # Updated region labels
        label_1 = Text("1", font_size=30, color=WHITE)
        label_2 = Text("2", font_size=30, color=WHITE)
        
        # Use grid alignment for labels to avoid overlap and confusion
        self.place_at_grid(label_1, 'D3', scale_factor=0.6) # Resolves Issue 20
        self.place_at_grid(label_2, 'B4', scale_factor=0.6) # Resolves Issue 21
        
        self.play(FadeIn(p2))
        self.play(Create(chord))
        self.play(
            FadeOut(initial_label),
            FadeIn(label_1),
            Write(label_2)
        )
        self.wait(2)
        
        # Reset colors
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
