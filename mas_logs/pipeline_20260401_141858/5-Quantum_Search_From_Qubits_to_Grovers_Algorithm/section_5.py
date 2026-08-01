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
        # Setup layout
        lecture_lines_text = [
            "The Diffusion Operator performs inversion about the mean.",
            "First, it calculates the average of all amplitudes.",
            "Then, it reflects every amplitude across that average level.",
            "The negative target amplitude is boosted significantly.",
            "This trampoline effect amplifies the correct answer's probability."
        ]
        self.setup_layout("Grover's Step 2: The Diffusion Operator", lecture_lines_text)

        # Colors
        CYAN_C = "#00BFFF"
        TOMATO_C = "#FF6347"
        GOLD_C = "#FFD700"
        WHITE_C = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(CYAN_C)
        
        # Constants for Bar Chart
        bar_width = 0.5
        initial_vals = [1.0, 1.0, -1.0, 1.0] # Relative amplitudes
        
        # Create bars and baseline
        baseline = Line(start=LEFT*2.2, end=RIGHT*2.2, color=GREY_E)
        
        bars = VGroup()
        for i, val in enumerate(initial_vals):
            color = CYAN_C if val > 0 else TOMATO_C
            rect = Rectangle(width=bar_width, height=abs(val), fill_color=color, fill_opacity=0.8, stroke_width=2)
            # Initial positioning relative to baseline center
            if val > 0:
                rect.next_to(baseline.point_from_proportion(0.2 + 0.2 * i), UP, buff=0)
            else:
                rect.next_to(baseline.point_from_proportion(0.2 + 0.2 * i), DOWN, buff=0)
            bars.add(rect)

        chart_group = VGroup(baseline, bars)
        # Use place_at_grid to stabilize the layout (Fixes Issue 46, 47, 48, 56)
        self.place_at_grid(chart_group, 'D4', scale_factor=1.1)
        
        self.play(Create(baseline), Create(bars))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(WHITE_C)
        
        # Calculate mean
        mean_val = sum(initial_vals) / len(initial_vals) # (1+1-1+1)/4 = 0.5
        # unit_height is visual height of bar with val 1.0 (after group scaling)
        unit_height = bars[0].get_height() / abs(initial_vals[0])
        
        mean_line = DashedLine(
            start=baseline.get_left(),
            end=baseline.get_right(),
            color=WHITE_C,
            stroke_width=2
        )
        mean_line.move_to(baseline.get_center() + UP * (mean_val * unit_height))
        
        mean_label = Text("Mean", font_size=24, color=WHITE_C)
        mean_label.next_to(mean_line, RIGHT, buff=0.2)
        
        self.play(Create(mean_line), Write(mean_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(WHITE_C)
        
        # Asset: Trampoline [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/trampoline.svg]
        trampoline = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/trampoline.svg")
        trampoline.set_color(WHITE_C)
        trampoline.set_width(1.0)
        trampoline.move_to(mean_line.get_center())
        
        self.play(FadeIn(trampoline))
        
        # Reflection math: new_val = 2 * mean - old_val
        # Results: [0, 0, 2, 0]
        target_vals = [2*mean_val - v for v in initial_vals]
        
        ref_anims = []
        for i, bar in enumerate(bars):
            v_target = target_vals[i]
            h_target = abs(v_target) * unit_height
            if h_target < 0.01: h_target = 0.01 # Keep visual baseline presence
            
            if v_target >= 0:
                ref_anims.append(bar.animate.stretch_to_fit_height(h_target).next_to(baseline.point_from_proportion(0.2 + 0.2 * i), UP, buff=0))
            else:
                ref_anims.append(bar.animate.stretch_to_fit_height(h_target).next_to(baseline.point_from_proportion(0.2 + 0.2 * i), DOWN, buff=0))
                
        self.play(*ref_anims, run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(GOLD_C)
        
        # Target bar is index 2. It has reflected to max height. Turn it gold.
        self.play(
            bars[2].animate.set_fill(GOLD_C).set_stroke(GOLD_C),
            FadeOut(trampoline)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(CYAN_C)
        
        # Clean up mean indicators and finalize shrunk non-targets
        shrink_anims = []
        for i in [0, 1, 3]:
            shrink_anims.append(bars[i].animate.stretch_to_fit_height(0.001).move_to(baseline.point_from_proportion(0.2 + 0.2 * i)))
            
        self.play(
            *shrink_anims,
            FadeOut(mean_line),
            FadeOut(mean_label)
        )
        self.wait(2)
