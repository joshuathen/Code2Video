from manim import *
import numpy as np

# Fix for KeyError: 'iπ' - Manim's config.get_dir method attempts to format file paths.
# The internal get_dir method uses a 'while "{" in path:' loop which recursively unescapes braces.
# To prevent the KeyError, we remove curly braces from the file path in the config dictionary.
if "input_file" in config._d:
    config._d["input_file"] = str(config._d["input_file"]).replace("{", "").replace("}", "")

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
        # Initialize Layout
        lecture_lines = [
            "Meet the most famous constants in mathematics.",
            "e, i, pi, 1, and 0 join together.",
            "This is Euler's identity: e^{i*pi} + 1 = 0.",
            "It unites analysis, algebra, and geometry beautifully.",
            "Five different worlds meet in one simple equation."
        ]
        self.setup_layout("The Grand Introduction: Meeting the Cast", lecture_lines)

        # Create constants mobjects using Text instead of MathTex to avoid LaTeX dependency
        # slant=ITALIC is used for mathematical variables to mimic LaTeX style
        e_mob = Text("e", slant=ITALIC, color=WHITE)
        i_mob = Text("i", slant=ITALIC, color=WHITE)
        pi_mob = Text("π", slant=ITALIC, color=WHITE)
        one_mob = Text("1", color=WHITE)
        zero_mob = Text("0", color=WHITE)
        
        plus_mob = Text("+", color=WHITE)
        equals_mob = Text("=", color=WHITE)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        self.place_at_grid(e_mob, "B2", scale_factor=1.5)
        self.place_at_grid(i_mob, "E1", scale_factor=1.5)
        self.place_at_grid(pi_mob, "B4", scale_factor=1.2) # Resolved Issue 22: Moved from A4 to B4
        self.place_at_grid(one_mob, "F3", scale_factor=1.5)
        self.place_at_grid(zero_mob, "B6", scale_factor=1.5)

        self.play(
            FadeIn(e_mob), FadeIn(i_mob), FadeIn(pi_mob), FadeIn(one_mob), FadeIn(zero_mob),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Target coordinates based on the grid for horizontal alignment
        target_e = self.grid["C1"]
        target_i = self.grid["C1"] + RIGHT * 0.45 + UP * 0.35
        target_pi = self.grid["C1"] + RIGHT * 0.85 + UP * 0.35
        target_plus = self.grid["C2"] + RIGHT * 0.5
        target_one = self.grid["C3"] + RIGHT * 0.5
        target_equals = self.grid["C4"] + RIGHT * 0.5
        target_zero = self.grid["C5"] + RIGHT * 0.5

        plus_mob.move_to(target_plus).scale(1.2)
        equals_mob.move_to(target_equals).scale(1.2)

        self.play(
            e_mob.animate.move_to(target_e).scale(0.8),
            i_mob.animate.move_to(target_i).scale(0.5),
            pi_mob.animate.move_to(target_pi).scale(0.5),
            one_mob.animate.move_to(target_one).scale(0.8),
            zero_mob.animate.move_to(target_zero).scale(0.8),
            FadeIn(plus_mob),
            FadeIn(equals_mob),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Glow effect group
        equation_group = VGroup(e_mob, i_mob, pi_mob, plus_mob, one_mob, equals_mob, zero_mob)
        
        self.play(
            equation_group.animate.set_color(WHITE),
            Flash(equation_group, color=WHITE, line_length=0.3, num_lines=12),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)

        # Labels
        analysis_label = Text("Analysis", font_size=18, color="#00FFFF")
        algebra_label = Text("Algebra", font_size=18, color="#FF00FF")
        geometry_label = Text("Geometry", font_size=18, color="#FFFF00")

        self.place_at_grid(analysis_label, "D1", scale_factor=1.0)
        self.place_at_grid(algebra_label, "B3", scale_factor=1.0) # Resolved Issue 21: Moved from B1 to B3
        self.place_at_grid(geometry_label, "D3", scale_factor=1.0) # Resolved Issue 20: Moved from D2 to D3

        # Highlight sequence
        self.play(
            e_mob.animate.set_color("#00FFFF"),
            FadeIn(analysis_label),
            run_time=1
        )
        self.play(
            i_mob.animate.set_color("#FF00FF"),
            FadeIn(algebra_label),
            run_time=1
        )
        self.play(
            pi_mob.animate.set_color("#FFFF00"),
            FadeIn(geometry_label),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)

        # Final emphasis - whole equation pulse
        self.play(
            equation_group.animate.scale(1.1),
            rate_func=there_and_back,
            run_time=2
        )
        self.wait(2)
