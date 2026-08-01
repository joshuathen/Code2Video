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
        # Define lecture lines
        lecture_lines = [
            'The Zeta function defines an infinite power sum.',
            'Variable s determines how fast terms shrink.',
            'Inputting s equals two yields pi squared over six.',
            'At s equals one, the series overflows to infinity.',
            'This boundary defines where the sum converges.'
        ]
        
        self.setup_layout("Defining the Zeta Function: The Summation Rule", lecture_lines)

        # Colors
        COLOR_ZETA = "#FFFF00"
        COLOR_EXPANSION = "#FFFFFF"
        COLOR_FACTORY = "#00FFFF"
        COLOR_BASEL = "#FFD700"
        COLOR_DIVERGENT = "#FF4500"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_ZETA)
        zeta_formula = Text("ζ(s) = Σ 1/n^s", color=COLOR_ZETA)
        self.place_in_area(zeta_formula, 'A2', 'A5', scale_factor=1.2)
        self.play(Write(zeta_formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_EXPANSION)
        expansion = Text(
            "= 1/1^s + 1/2^s + 1/3^s + ...", 
            color=COLOR_EXPANSION
        )
        self.place_in_area(expansion, 'B2', 'B5', scale_factor=0.9)
        self.play(FadeIn(expansion, shift=DOWN))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_BASEL)
        
        # Number Factory setup
        factory_rect = Rectangle(width=3.5, height=1.5, color=COLOR_FACTORY, fill_opacity=0.2)
        factory_text = Text("Number Factory", font_size=20, color=COLOR_FACTORY)
        factory = VGroup(factory_rect, factory_text)
        # Fix Issue 37: Reposition factory to avoid gaps and overcrowding
        self.place_in_area(factory, 'C2', 'C5', scale_factor=0.8)
        
        # Adjust arrows to row C to match new factory position
        input_arrow = Arrow(start=self.grid['C1'], end=self.grid['C2'], color=COLOR_FACTORY)
        input_label = Text("s = 2", color=COLOR_FACTORY)
        input_label.next_to(input_arrow, UP, buff=0.1)
        
        output_arrow = Arrow(start=self.grid['C5'], end=self.grid['C6'], color=COLOR_BASEL)
        output_val = Text("π² / 6", color=COLOR_BASEL)
        output_val.next_to(output_arrow, RIGHT, buff=0.1)
        
        basel_eq = Text("Σ 1/n² = π² / 6", color=COLOR_BASEL)
        # Fix Issue 38: Move Basel equation to Row E for better flow
        self.place_in_area(basel_eq, 'E2', 'E5', scale_factor=1.0)

        self.play(FadeIn(factory))
        self.play(GrowArrow(input_arrow), FadeIn(input_label))
        
        # Factory Flash and Completion of Line 3
        self.play(Flash(factory_rect, color=COLOR_FACTORY))
        self.play(GrowArrow(output_arrow), FadeIn(output_val))
        self.play(Write(basel_eq))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(COLOR_DIVERGENT)
        s_one_label = Text("s = 1", color=COLOR_DIVERGENT).move_to(input_label)
        inf_val = Text("∞", color=COLOR_DIVERGENT).move_to(output_val)
        harmonic_eq = Text("Σ 1/n¹ = ∞", color=COLOR_DIVERGENT)
        # Fix Issue 39: Move harmonic equation to Row E for consistency
        self.place_in_area(harmonic_eq, 'E2', 'E5', scale_factor=1.0)

        self.play(
            Transform(input_label, s_one_label),
            FadeOut(output_val),
            FadeOut(basel_eq),
            Flash(factory_rect, color=COLOR_DIVERGENT)
        )
        self.play(FadeIn(inf_val), Write(harmonic_eq))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(WHITE)
        highlight = SurroundingRectangle(harmonic_eq, color=WHITE)
        self.play(Create(highlight))
        self.play(Indicate(self.lecture[4]))
        self.wait(2)
