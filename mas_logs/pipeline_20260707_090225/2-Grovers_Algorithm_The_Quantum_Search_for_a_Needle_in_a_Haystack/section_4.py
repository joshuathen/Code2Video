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
        # Setup standard layout
        lecture_lines = [
            'Next, we apply the Grover diffusion operator.',
            'We calculate the average amplitude of all states.',
            'Every amplitude is reflected across this average line.',
            "The target's amplitude grows significantly taller than others.",
            'The non-target amplitudes shrink slightly in comparison.'
        ]
        self.setup_layout("Step 2: The Diffusion Operator (Reflection about the Mean)", lecture_lines)

        # Colors
        COLOR_TARGET = "#FFD700"  # Gold
        COLOR_NON_TARGET = "#50C878"  # Emerald Green
        COLOR_MEAN = "#FFFFFF"  # White
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Create a set of bars representing amplitudes after the Oracle step
        # Target is at index 2, flipped negative
        initial_heights = [1.2, 1.2, -1.2, 1.2, 1.2, 1.2, 1.2, 1.2]
        bar_colors = [COLOR_NON_TARGET] * 8
        bar_colors[2] = COLOR_TARGET
        
        bars = VGroup()
        bar_width = 0.4
        spacing = 0.1
        
        for i, h in enumerate(initial_heights):
            rect = Rectangle(
                width=bar_width, 
                height=abs(h), 
                fill_opacity=0.8, 
                fill_color=bar_colors[i],
                stroke_width=1
            )
            if h >= 0:
                rect.align_to(ORIGIN, DOWN)
            else:
                rect.align_to(ORIGIN, UP)
            
            rect.shift(RIGHT * i * (bar_width + spacing))
            bars.add(rect)
            
        baseline = Line(
            bars.get_left() + LEFT * 0.2, 
            bars.get_right() + RIGHT * 0.2, 
            color=GRAY, 
            stroke_width=2
        )
        chart = VGroup(baseline, bars)
        
        # [Issue 30 Fix]: Updated placement area and scale
        self.place_in_area(chart, 'B2', 'E5', scale_factor=1.2)
        
        self.play(FadeIn(chart))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Calculate mean: (7 * 1.2 - 1.2) / 8 = 7.2 / 8 = 0.9
        mean_val = sum(initial_heights) / len(initial_heights)
        mean_y = baseline.get_y() + mean_val * (bars[0].height / abs(initial_heights[0]))
        
        mean_line = DashedLine(
            start=[baseline.get_left()[0], mean_y, 0],
            end=[baseline.get_right()[0], mean_y, 0],
            color=COLOR_MEAN,
            stroke_width=3
        )
        
        # [Issue 31 Fix]: Updated placement for label
        mean_label = Text("Mean", font_size=16, color=COLOR_MEAN)
        self.place_at_grid(mean_label, 'C6', scale_factor=0.8)
        
        self.play(Create(mean_line), Write(mean_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Reflection logic: H_new = 2 * Mean - H_old
        new_heights = [2 * mean_val - h for h in initial_heights]
        
        new_bars = VGroup()
        for i, (h_old, h_new) in enumerate(zip(initial_heights, new_heights)):
            new_rect = Rectangle(
                width=bar_width * 1.2, # Matching chart scale
                height=abs(h_new) * 1.2, # Matching chart scale
                fill_opacity=0.8, 
                fill_color=bar_colors[i],
                stroke_width=1
            )
            new_rect.move_to(bars[i].get_center())
            new_rect.set_x(bars[i].get_x())
            
            if h_new >= 0:
                new_rect.align_to(baseline, DOWN)
            else:
                new_rect.align_to(baseline, UP)
            
            new_bars.add(new_rect)
            
        self.play(
            ReplacementTransform(bars, new_bars),
            run_time=2,
            rate_func=slow_into
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        target_bar = new_bars[2]
        
        # [Issue 32 Fix]: Anchor target bar expansion to specific grid row area
        # Calculate destination center for B4-E4
        tl_pos = self.grid['B4']
        br_pos = self.grid['E4']
        target_center = np.array([(tl_pos[0] + br_pos[0]) / 2, (tl_pos[1] + br_pos[1]) / 2, 0])
        
        surround_rect = SurroundingRectangle(target_bar, color=COLOR_TARGET, buff=0.1)
        
        self.play(
            Create(surround_rect),
            target_bar.animate.move_to(target_center).scale(1.0)
        )
        self.play(Indicate(target_bar, color=COLOR_TARGET))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        non_targets = VGroup(*[new_bars[i] for i in range(len(new_bars)) if i != 2])
        self.play(FadeOut(surround_rect))
        self.play(non_targets.animate.set_fill(opacity=0.4))
        self.play(Indicate(non_targets))
        
        self.wait(2)
        self.lecture[4].set_color(WHITE)
