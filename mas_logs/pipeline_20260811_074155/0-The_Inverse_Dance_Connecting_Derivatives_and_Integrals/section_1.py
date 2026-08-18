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
        # Setup the layout
        title_text = "Introduction: Meet Robby the Robot"
        lecture_lines = [
            "Meet Robby, our robotic guide to the world of calculus.",
            "Derivatives track Robby's speed at any single moment.",
            "Integrals measure the total distance Robby has traveled.",
            "One looks at the 'now', the other the 'sum'.",
            "Together, they reveal the hidden harmony of change."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Define objects early to avoid recreating in always_redraw
        # Load Assets (Issue 19)
        robby = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg").set_color("#87CEEB")
        
        speedometer_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speed.svg").set_color("#FFFF00")
        vt_label = MathTex("v(t)", color="#FFFF00", font_size=24)
        speedometer = VGroup(speedometer_asset, vt_label).arrange(DOWN, buff=0.1)

        # Distance Bar
        bar_bg = Rectangle(width=4.0, height=0.3, color=WHITE)
        bar_fill = Rectangle(width=0.01, height=0.25, fill_opacity=1, color="#00FF00", stroke_width=0)
        bar_fill.align_to(bar_bg, LEFT).shift(RIGHT*0.05)
        dist_label = Text("Total Distance", color="#00FF00", font_size=20).next_to(bar_bg, DOWN, buff=0.1)
        distance_bar = VGroup(bar_bg, bar_fill, dist_label)

        # Labels
        roc_label = Text("Rate of Change", color="#FFFF00", font_size=22)
        acc_label = Text("Accumulation", color="#00FF00", font_size=22)

        # Final Text
        final_text = Text("Calculus: The Two Sides", color=WHITE, font_size=36)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#87CEEB")
        # Issue 23: Robby starts at D4
        self.place_at_grid(robby, "D4", scale_factor=0.6)
        self.play(FadeIn(robby))
        # Robby walks horizontally across
        self.play(robby.animate.move_to(self.grid["D6"]), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFF00")
        # Issue 24: speedometer at B4
        self.place_at_grid(speedometer, "B4", scale_factor=0.8)
        # Move Robby to be under the speedometer position (D4)
        self.play(robby.animate.move_to(self.grid["D4"]))
        self.play(FadeIn(speedometer))
        # Simple pulse animation for speed activity
        self.play(speedometer.animate.scale(1.1), run_time=0.5)
        self.play(speedometer.animate.scale(1/1.1), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FF00")
        # Distance Bar fills in green
        self.place_in_area(distance_bar, "F1", "F6", scale_factor=1.0)
        self.play(FadeIn(bar_bg), FadeIn(dist_label))
        self.play(
            bar_fill.animate.stretch_to_fit_width(3.9).align_to(bar_bg, LEFT).shift(RIGHT*0.05),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#FF8C00")
        # Issue 24: roc_label at A4, acc_label at E4
        self.place_at_grid(roc_label, "A4")
        self.place_at_grid(acc_label, "E4")
        self.play(Write(roc_label), Write(acc_label))
        self.play(Indicate(speedometer), Indicate(distance_bar))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#FF69B4")
        # Issue 25: final_text at B2 to E5
        self.place_in_area(final_text, "B2", "E5")
        self.play(
            FadeOut(robby), FadeOut(speedometer), FadeOut(distance_bar),
            FadeOut(roc_label), FadeOut(acc_label)
        )
        self.play(Write(final_text))
        self.wait(2)
