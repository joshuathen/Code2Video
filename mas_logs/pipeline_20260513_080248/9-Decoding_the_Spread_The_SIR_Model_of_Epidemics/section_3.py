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
        # Setup layout
        lecture_lines = [
            'The population is split into three distinct compartments.', 
            'Healthy individuals begin in the Susceptible group.', 
            'They become Infectious after contacting the virus.', 
            'Eventually, they recover and gain lasting immunity.', 
            'The total population remains constant across all groups.'
        ]
        self.setup_layout("The SIR Compartments", lecture_lines)

        # Colors
        COLOR_S = "#3498db" # Blue
        COLOR_I = "#e74c3c" # Red
        COLOR_R = "#2ecc71" # Green
        COLOR_TEXT = "#ffffff"
        ASSET_PATH = "/mmfs1/data/home/jthen/Code2Video/assets/icon/person.svg"

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1
        self.play(self.lecture[0].animate.set_color(YELLOW))

        # Create Compartment Rectangles
        s_rect = Rectangle(width=1.6, height=2.6, color=COLOR_S, stroke_width=4)
        i_rect = Rectangle(width=1.6, height=2.6, color=COLOR_I, stroke_width=4)
        r_rect = Rectangle(width=1.6, height=2.6, color=COLOR_R, stroke_width=4)

        self.place_in_area(s_rect, "B1", "D2")
        self.place_in_area(i_rect, "B3", "D4")
        self.place_in_area(r_rect, "B5", "D6")

        # Labels - Addressing Issues 43, 44, 45 (Centered under compartments)
        s_label = Text("Susceptible", font_size=18, color=COLOR_S)
        i_label = Text("Infectious", font_size=18, color=COLOR_I)
        r_label = Text("Recovered", font_size=18, color=COLOR_R)

        self.place_in_area(s_label, 'E1', 'E2', scale_factor=0.8)
        self.place_in_area(i_label, 'E3', 'E4', scale_factor=0.8)
        self.place_in_area(r_label, 'E5', 'E6', scale_factor=0.8)

        self.play(
            Create(s_rect), Create(i_rect), Create(r_rect),
            Write(s_label), Write(i_label), Write(r_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition highlight
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )

        # Populate Susceptible zone with 10 person icons [Asset: ...]
        people = VGroup()
        for i in range(10):
            person = SVGMobject(ASSET_PATH).set_color(COLOR_S)
            person.scale(0.12)
            # Scatter within S rect (B1-D2)
            row = i // 2
            col = i % 2
            # Manual jitter/offset within the box center
            offset = np.array([(col - 0.5) * 0.5, (1.0 - row * 0.5), 0])
            person.move_to(s_rect.get_center() + offset)
            people.add(person)

        # Equation display N = S + I + R
        equation = Text("N = S + I + R", font_size=32, color=COLOR_TEXT)
        self.place_in_area(equation, "A1", "A6")

        self.play(
            LaggedStart(*[FadeIn(p) for p in people], lag_ratio=0.05),
            Write(equation)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transition highlight
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )

        # Animate one person icon moving S -> I (turning red)
        traveler = people[0]
        self.play(
            traveler.animate.move_to(i_rect.get_center()).set_color(COLOR_I),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Transition highlight
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW)
        )

        # Animate icon moving I -> R (turning green)
        self.play(
            traveler.animate.move_to(r_rect.get_center()).set_color(COLOR_R),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Transition highlight
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )
        
        # Emphasize the conservation equation
        self.play(equation.animate.set_color(YELLOW).scale(1.1))
        self.play(equation.animate.set_color(WHITE).scale(1/1.1))

        # Reset line color
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(2)
