from manim import *
import numpy as np

# Fix for potential KeyError in config with file paths containing braces
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

class Section6Scene(TeachingScene):
    def construct(self):
        # 1. Setup Layout
        lecture_lines = [
            "We start at positive one on the real line.",
            "Traveling halfway around, we reach the exact opposite side.",
            "Our destination is precisely at negative one.",
            "The reunion is complete: e, i, and pi align.",
            "Euler's identity: the most beautiful equation in math."
        ]
        self.setup_layout("The Destination: e^{iπ} = -1", lecture_lines)

        # Colors for constants
        E_COLOR = "#FF00FF"
        I_COLOR = "#00FFFF"
        PI_COLOR = "#FFFF00"
        ONE_COLOR = "#00FF00"
        ZERO_COLOR = "#FFFFFF"
        TEXT_GOLD = "#FFD700"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE_C))
        
        # Start point at '1' (D5)
        point_pos_one = Dot(color=ONE_COLOR, radius=0.1)
        self.place_at_grid(point_pos_one, "D5", scale_factor=1.0)
        label_pos_one = Text("1", font_size=16).next_to(point_pos_one, DOWN, buff=0.1)
        
        self.play(FadeIn(point_pos_one), Write(label_pos_one))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE_C))
        
        # Arc representing the path. Mandatory positioning in B2-E5
        # We create a semi-circle that will visually bridge D5 and D2
        arc_path = Arc(radius=1.5, start_angle=0, angle=PI, color=WHITE)
        self.place_in_area(arc_path, 'B2', 'E5', scale_factor=0.7)
        
        self.play(Create(arc_path), run_time=2)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(BLUE_C))
        
        # Destination point at negative one. Mandatory positioning at D2
        point_neg_one = Dot(color="#FF0000", radius=0.15)
        self.place_at_grid(point_neg_one, 'D2', scale_factor=0.8)
        label_neg_one = Text("-1", font_size=16).next_to(point_neg_one, DOWN, buff=0.1)
        
        self.play(FadeIn(point_neg_one), Write(label_neg_one))
        self.play(Flash(point_neg_one, color="#FF0000", line_length=0.3, num_lines=12, flash_radius=0.3))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(BLUE_C))
        
        # Write equation e^{i*pi} = -1. Mandatory positioning in F2-F5
        e_text = Text("e", color=E_COLOR, font_size=36)
        i_text = Text("i", color=I_COLOR, font_size=24)
        pi_text = Text("π", color=PI_COLOR, font_size=24)
        
        exp_group = VGroup(i_text, pi_text).arrange(RIGHT, buff=0.05).scale(0.8)
        exp_group.next_to(e_text.get_top(), RIGHT, buff=-0.1).shift(UP*0.1)
        lhs = VGroup(e_text, exp_group)
        
        eq_sign = Text(" = ", color=WHITE, font_size=36)
        minus_sign = Text("-", color=WHITE, font_size=36)
        one_text = Text("1", color=ONE_COLOR, font_size=36)
        
        euler_identity = VGroup(lhs, eq_sign, minus_sign, one_text).arrange(RIGHT, buff=0.2)
        self.place_in_area(euler_identity, 'F2', 'F5', scale_factor=0.9)
        
        self.play(Write(euler_identity), run_time=2.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(BLUE_C))
        
        # Transform equation to e^{i*pi} + 1 = 0
        plus_sign = Text("+", color=WHITE, font_size=36)
        zero_text = Text("0", color=ZERO_COLOR, font_size=36)
        
        new_euler_identity = VGroup(lhs.copy(), plus_sign, one_text.copy(), eq_sign.copy(), zero_text).arrange(RIGHT, buff=0.2)
        self.place_in_area(new_euler_identity, 'F2', 'F5', scale_factor=0.9)
        
        self.play(ReplacementTransform(euler_identity, new_euler_identity))
        
        # Pulsing effect and glow constant color markers
        self.play(
            new_euler_identity.animate.scale(1.2),
            run_time=0.5, rate_func=there_and_back
        )
        self.wait(0.5)
        
        # Final display: 'The Most Beautiful Equation' + landmark.svg
        landmark_asset = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/landmark.svg")
        self.place_at_grid(landmark_asset, "B4", scale_factor=0.6)
        
        final_text = Text("The Most Beautiful Equation", color=TEXT_GOLD, font_size=24)
        self.place_at_grid(final_text, "A4", scale_factor=1.0)
        
        self.play(
            FadeIn(landmark_asset, shift=UP),
            Write(final_text),
            run_time=1.5
        )
        
        # Pulsing the final equation again
        self.play(Indicate(new_euler_identity, color=TEXT_GOLD))
        self.wait(3)
