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

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup layout with new teaching script
        lines = [
            "We repeat the Oracle and Diffusion steps multiple times.",
            "For one thousand items, we only need thirty-two iterations.",
            "Each pulse of light makes the target bar grow.",
            "The target's probability eventually nears one hundred percent.",
            "Finally, we measure the state to find the target."
        ]
        self.setup_layout("Iteration and Measurement", lines)

        # Define consistent colors for matching animations
        color_iter = YELLOW
        color_count = BLUE
        color_pulse = GREEN
        color_prob = ORANGE
        color_meas = WHITE

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(color_iter)
        
        # Iteration counter setup
        step_label = Text("Steps: ", font_size=30, color=color_iter)
        step_val = Integer(1, color=color_iter, mob_class=Text)
        counter = VGroup(step_label, step_val).arrange(RIGHT, buff=0.1)
        self.place_at_grid(counter, "A6", scale_factor=0.8)

        # Bar chart setup (5 bars representing the search space)
        bars = VGroup(*[
            Rectangle(width=0.5, height=0.2, fill_opacity=0.7, color=BLUE_B, stroke_width=1) 
            for _ in range(5)
        ]).arrange(RIGHT, buff=0.3)
        target_idx = 2
        bars[target_idx].set_color(GOLD)
        self.place_in_area(bars, "C2", "E5")
        
        baseline = Line(
            self.grid["E1"] + LEFT * 0.5 + DOWN * 0.5, 
            self.grid["E6"] + RIGHT * 0.5 + DOWN * 0.5, 
            color=GREY
        )
        # Manually align baseline to the bottom of the bars area
        bars_bottom = bars.get_bottom()[1]
        baseline.move_to([bars.get_center()[0], bars_bottom, 0])

        self.play(FadeIn(counter), Create(bars), Create(baseline))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(color_count)
        # Show iteration steps increasing
        self.play(step_val.animate.set_value(32), run_time=1.5, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(color_pulse)
        # Target bar grows while others shrink
        # Pulse effect (Oracle/Diffusion)
        pulse = Circle(radius=0.1, color=color_pulse, stroke_width=4).move_to(bars[target_idx])
        
        self.play(
            pulse.animate.scale(10).set_stroke(opacity=0),
            bars[target_idx].animate.stretch_to_fit_height(3.0, about_edge=DOWN).set_fill(color_pulse, opacity=0.9),
            *[b.animate.stretch_to_fit_height(0.05, about_edge=DOWN) for i, b in enumerate(bars) if i != target_idx],
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(color_prob)
        # Probability label at safe grid coordinate D6 as suggested by issue fix logic
        prob_label = Text("Prob: 99.9%", font_size=24, color=color_prob)
        self.place_at_grid(prob_label, "D6", scale_factor=0.9)
        
        self.play(Write(prob_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(color_meas)
        # Measurement flash on the target bar
        flash_rect = Rectangle(
            width=bars[target_idx].width * 1.5,
            height=bars[target_idx].height * 1.1,
            color=WHITE,
            fill_opacity=0.3,
            stroke_width=0
        ).move_to(bars[target_idx])
        
        flash_effect = Flash(
            bars[target_idx].get_top(), 
            color=WHITE, 
            line_length=0.5, 
            flash_radius=0.6,
            num_lines=12
        )

        self.play(
            FadeIn(flash_rect),
            flash_effect
        )
        self.play(FadeOut(flash_rect))
        self.wait(2)
