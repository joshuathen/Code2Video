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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "Backpropagation: The Blame Game", 
            [
                "Backpropagation traces the error back through the network.",
                "We calculate how much each weight contributed to error.",
                "The chain rule helps calculate these sensitivities precisely.",
                "This assigns \"blame\" to specific dials in the system.",
                "We now know which direction to turn each dial."
            ]
        )
        
        # Elements
        error_box = RoundedRectangle(corner_radius=0.1, height=1.0, width=1.5, color="#FF0000")
        error_text = Text("Error Score", font_size=18, color="#FF0000")
        error_group = VGroup(error_box, error_text)
        # Fix Issue 28: Reposition error_group
        self.place_in_area(error_group, 'B5', 'D6', scale_factor=0.8)

        # Assets integration (Issue 20)
        # Using timer.svg for Timer and dial.svg for Temp
        timer_dial = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/timer.svg", color=WHITE)
        timer_label = Text("Timer", font_size=16, color=WHITE).next_to(timer_dial, DOWN, buff=0.1)
        timer_group = VGroup(timer_dial, timer_label)
        self.place_at_grid(timer_group, "B2", scale_factor=0.6)

        temp_dial = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dial.svg", color=WHITE)
        temp_label = Text("Temp", font_size=16, color=WHITE).next_to(temp_dial, DOWN, buff=0.1)
        temp_group = VGroup(temp_dial, temp_label)
        self.place_at_grid(temp_group, "D2", scale_factor=0.6)

        # Symbols for sensitivities
        timer_grad = MathTex(r"\frac{\partial E}{\partial w_{timer}}", font_size=24, color="#00FFFF")
        temp_grad = MathTex(r"\frac{\partial E}{\partial w_{temp}}", font_size=24, color="#00FFFF")
        # Fix Issue 29 & 30: Reposition gradient labels
        self.place_at_grid(timer_grad, 'A3', scale_factor=0.7)
        self.place_at_grid(temp_grad, 'C3', scale_factor=0.7)

        # Arrows (Backwards)
        # Pointing from error_group towards the dials
        arrow1 = Arrow(start=error_group.get_left(), end=timer_group.get_right(), buff=0.2, color=WHITE)
        arrow2 = Arrow(start=error_group.get_left(), end=temp_group.get_right(), buff=0.2, color=WHITE)

        # Highlight Border for the highest error contribution (Timer)
        # Use a surrounding rectangle for the glowing border (Issue 20)
        glowing_border = SurroundingRectangle(timer_dial, color="#FFFFFF", buff=0.1, stroke_width=4)

        # === Animation for Lecture Line 1 ===
        # "Backpropagation traces the error back through the network."
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(FadeIn(error_group))
        
        # Flashing effect for Error Score (Animation 1)
        # Using a ValueTracker to avoid expensive re-creation in always_redraw if possible, 
        # but here we just animate the opacity of the existing box.
        self.play(error_box.animate.set_fill(opacity=0.3), run_time=0.5)
        self.play(error_box.animate.set_fill(opacity=0.0), run_time=0.5)
        self.play(error_box.animate.set_fill(opacity=0.3), run_time=0.5)
        self.play(error_box.animate.set_fill(opacity=0.0), run_time=0.5)

        # === Animation for Lecture Line 2 ===
        # "We calculate how much each weight contributed to error."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        self.play(FadeIn(timer_group), FadeIn(temp_group))
        self.play(GrowArrow(arrow1), GrowArrow(arrow2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "The chain rule helps calculate these sensitivities precisely."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        self.play(Write(timer_grad), Write(temp_grad))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "This assigns \"blame\" to specific dials in the system."
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW)
        )
        # Timer Weight: #8B0000, Temp Weight: #FA8072
        self.play(
            timer_dial.animate.set_color("#8B0000"),
            timer_label.animate.set_color("#8B0000"),
            temp_dial.animate.set_color("#FA8072"),
            temp_label.animate.set_color("#FA8072")
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "We now know which direction to turn each dial."
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )
        # Highlight highest contributor (Timer) with Asset: dial.svg or just the glowing border
        # Storyboard says: Highlight the weight [Asset: dial.svg] with the highest error contribution.
        # Since I used timer.svg for timer and dial.svg for temp, but timer has higher blame, 
        # I'll highlight the timer.
        self.play(Create(glowing_border))
        self.play(Indicate(timer_dial, color="#FFFFFF"))
        self.wait(2)
        
        # Reset colors for final state
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
