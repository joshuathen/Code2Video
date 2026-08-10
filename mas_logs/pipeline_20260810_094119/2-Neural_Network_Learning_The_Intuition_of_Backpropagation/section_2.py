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
        self.setup_layout("The Prerequisite: The Weight 'Knobs'", [
            "Weights are like adjustable knobs on a machine.",
            "Each knob changes the final system output.",
            "Learning is tuning these knobs for perfection."
        ])
        
        # Assets
        knob_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/knob.svg"
        hand_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/hand.svg"
        
        knob = SVGMobject(knob_path)
        hand = SVGMobject(hand_path)
        weight_label = MathTex("Weight", font_size=24)
        knob_group = VGroup(knob, weight_label).arrange(DOWN)
        
        conn_line = Line(start=UP, end=DOWN, color="#00FFFF", stroke_width=4)
        output_eq = MathTex("Output = Input * Weight", color="#00FFFF", font_size=28)
        signal_indicator = Circle(radius=0.3, color="#FF9900", fill_opacity=0.5)
        
        knobs_cluster = VGroup(*[SVGMobject(knob_path).scale(0.3) for _ in range(5)]).arrange(RIGHT)

        # Positioning (using grid fixes from issues 24, 25, 26)
        self.place_at_grid(knob_group, 'B4', scale_factor=0.6)
        self.place_at_grid(conn_line, 'C4', scale_factor=0.7)
        self.place_at_grid(signal_indicator, 'D5', scale_factor=0.7)
        self.place_at_grid(output_eq, 'E4', scale_factor=0.7)
        self.place_at_grid(knobs_cluster, 'F3', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"), FadeIn(knob_group), run_time=1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"), 
                  FadeIn(hand),
                  hand.animate.move_to(knob.get_center()),
                  run_time=1.0)
        self.play(Rotate(knob, angle=PI/4, about_point=knob.get_center()),
                  Rotate(knob, angle=-PI/2, about_point=knob.get_center()),
                  run_time=1.5)
        self.play(FadeOut(hand), Create(conn_line), Write(output_eq), run_time=1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF9900"), 
                  FadeIn(signal_indicator), 
                  FadeIn(knobs_cluster),
                  run_time=1.5)
        self.wait(1)
