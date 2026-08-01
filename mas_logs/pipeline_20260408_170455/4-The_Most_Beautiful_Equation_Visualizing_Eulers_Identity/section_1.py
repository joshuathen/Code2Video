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
        # Initial setup
        lecture_lines = [
            "This is the most beautiful equation in mathematics.",
            "It unites five fundamental constants in one simple line.",
            "Zero, one, e, i, and pi join together.",
            "From the void of zero to the growth of e.",
            "All connected in a single, elegant relationship."
        ]
        self.setup_layout("The Cosmic Reunion", lecture_lines)

        # Create equation components using Text to avoid LaTeX dependency (FileNotFoundError: latex)
        e_mob = Text("e", font_size=60)
        i_mob = Text("i", font_size=40)
        pi_mob = Text("π", font_size=40)
        plus_mob = Text("+", font_size=60)
        one_mob = Text("1", font_size=60)
        equal_mob = Text("=", font_size=60)
        zero_mob = Text("0", font_size=60)

        # Arrange exponent
        exponent = VGroup(i_mob, pi_mob).arrange(RIGHT, buff=0.05)
        exponent.next_to(e_mob, UR, buff=-0.1).shift(UP * 0.2)
        
        # Arrange full equation
        eq_right_part = VGroup(plus_mob, one_mob, equal_mob, zero_mob).arrange(RIGHT, buff=0.3)
        eq_right_part.next_to(e_mob, RIGHT, buff=0.5)
        
        full_equation = VGroup(e_mob, exponent, eq_right_part)
        
        # Use a horizontal area C1-D6 for the equation to allow better expansion and avoid oversized appearance
        self.place_in_area(full_equation, "C1", "D6", scale_factor=0.85)

        # === Animation for Lecture Line 1 ===
        # Display the full equation in white
        self.play(
            Write(full_equation),
            self.lecture[0].animate.set_color(WHITE),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Apply distinct colors to each constant
        # e (#55FF55), i (#5555FF), π (#FF5555), 1 (#FFFFFF), 0 (#AAAAAA)
        self.play(
            e_mob.animate.set_color("#55FF55"),
            i_mob.animate.set_color("#5555FF"),
            pi_mob.animate.set_color("#FF5555"),
            one_mob.animate.set_color("#FFFFFF"),
            zero_mob.animate.set_color("#AAAAAA"),
            self.lecture[1].animate.set_color(YELLOW),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Emphasize the connection
        self.play(
            Indicate(full_equation, color=WHITE),
            self.lecture[2].animate.set_color(WHITE),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Scale up 'e' and growth pulse effect
        self.play(
            e_mob.animate.scale(1.3).set_color("#55FF55"),
            self.lecture[3].animate.set_color("#55FF55"),
            run_time=0.8
        )
        self.play(
            e_mob.animate.scale(1/1.3),
            run_time=0.8
        )
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        # Scale up 'i' and 'π' as they glow and slightly vibrate
        # Golden glow for the entire equation
        self.play(
            exponent.animate.scale(1.2),
            self.lecture[4].animate.set_color("#FFFF88"),
            run_time=1
        )
        
        # Vibrate (Wiggle)
        self.play(
            Wiggle(exponent),
            run_time=1
        )

        # Final Golden Glow
        glow_rect = SurroundingRectangle(full_equation, color="#FFFF88", buff=0.4)
        self.play(
            Create(glow_rect),
            full_equation.animate.set_color("#FFFF88"),
            run_time=1.5
        )
        self.play(FadeOut(glow_rect), run_time=1)
        self.wait(3)
