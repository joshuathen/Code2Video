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
        lecture_lines = [
            "Neural networks are machines that learn from mistakes.",
            "Think of a student identifying photos of cats.",
            "The network makes a guess at the image.",
            "If wrong, it feels frustration as an error signal.",
            "It then adjusts its internal knobs to improve."
        ]
        self.setup_layout("The Analogy: Learning from Mistakes", lecture_lines)
        
        # Elements
        input_dot = Dot(color=WHITE)
        hidden_dot = Dot(color=WHITE)
        output_dot = Dot(color=WHITE)
        input_lbl = Text("Input", font_size=20)
        hidden_lbl = Text("Hidden", font_size=20)
        output_lbl = Text("Output", font_size=20)
        
        # Group and Place
        net = VGroup(
            VGroup(input_dot, input_lbl).arrange(DOWN),
            VGroup(hidden_dot, hidden_lbl).arrange(DOWN),
            VGroup(output_dot, output_lbl).arrange(DOWN)
        ).arrange(RIGHT, buff=1.0)
        
        # Placing per instructions (Issue 34)
        self.place_in_area(net, 'C4', 'E6', scale_factor=0.85)
        
        # Assets
        cat_icon = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png")
        knob_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/knob.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.play(FadeIn(net))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(GREEN)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        guess_label = Text("Guess: Cat", color="#FF0000", font_size=24)
        self.place_at_grid(guess_label, "B3")
        self.place_at_grid(cat_icon, "B4", scale_factor=0.5)
        
        arrow = Arrow(input_dot.get_center(), hidden_dot.get_center(), color=YELLOW)
        self.play(Create(arrow), FadeIn(guess_label), FadeIn(cat_icon))

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(RED)
        error_flash = Annulus(inner_radius=0.2, outer_radius=0.4, color=RED).move_to(output_dot.get_center())
        self.play(Flash(output_dot, color=RED), FadeIn(error_flash))

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(BLUE)
        
        knob_group = VGroup(knob_icon).set_color("#32CD32")
        self.place_at_grid(knob_group, "F3", scale_factor=0.6)
        
        self.play(FadeIn(knob_group), Indicate(net), FadeOut(error_flash))
        self.wait(1)
