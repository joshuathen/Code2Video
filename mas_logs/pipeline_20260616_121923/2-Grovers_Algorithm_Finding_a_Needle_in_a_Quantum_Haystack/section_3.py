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

class Section3Scene(TeachingScene):
    def construct(self):
        title = "Step 1: The Oracle (Phase Inversion)"
        lines = [
            'The Oracle identifies the target item within the system.',
            "It marks the solution by flipping its amplitude's sign.",
            'All other states remain unchanged with positive values.',
            'Now, the target bar points downwards on our graph.',
            'This phase inversion distinguishes the solution from the rest.'
        ]
        self.setup_layout(title, lines)

        # Colors
        GOLD = "#FFD700"
        WHITE_CLR = "#FFFFFF"

        # Create the bar graph elements
        bar_states = ["000", "001", "010", "011", "100", "101", "110", "111"]
        target_index = 5  # |101>
        
        bar_width = 0.4
        bar_height = 1.5
        
        bars = VGroup()
        labels = VGroup()
        
        for i, state in enumerate(bar_states):
            bar = Rectangle(
                width=bar_width, 
                height=bar_height, 
                fill_opacity=0.8, 
                fill_color=WHITE_CLR, 
                stroke_width=1
            )
            # Relative positioning within group
            bar.move_to(RIGHT * i * 0.6)
            bars.add(bar)
            
            label = Text(f"|{state}>", font_size=18, color=WHITE_CLR)
            # Significant buffer to allow for inverted bars without occlusion
            label.next_to(bar, DOWN, buff=2.2) 
            labels.add(label)

        graph_group = VGroup(bars, labels)
        # Issue 34 fix: scale factor 0.75
        self.place_in_area(graph_group, "B1", "E6", scale_factor=0.75)
        
        # Baseline
        baseline = Line(
            start=bars[0].get_corner(DL) + LEFT * 0.2,
            end=bars[-1].get_corner(DR) + RIGHT * 0.2,
            color=WHITE_CLR
        )
        self.add(baseline)
        
        # Initialize bars to sit ON the baseline
        for bar in bars:
            bar.align_to(baseline, DOWN)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(GOLD)
        self.play(Create(bars), Create(labels), run_time=1.5)
        self.wait(0.5)
        
        # Highlight target state |101> in gold
        self.play(
            bars[target_index].animate.set_color(GOLD),
            labels[target_index].animate.set_color(GOLD),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE_CLR)
        self.lecture[1].set_color(GOLD)
        
        # Issue 33: Oracle label at A5
        oracle_label = Text("Oracle", font_size=24, color=GOLD)
        self.place_at_grid(oracle_label, "A5", scale_factor=1.0)
        
        # Issue 27: Oracle icon [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/oracle.svg]
        oracle_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/oracle.svg")
        self.place_at_grid(oracle_icon, "A4", scale_factor=0.7)
        
        # Flipping animation
        self.play(
            FadeIn(oracle_icon),
            Write(oracle_label),
            bars[target_index].animate.stretch_to_fit_height(bar_height, about_edge=UP).align_to(baseline, UP),
            run_time=1.5
        )
        
        # Pulse effect around the inverted bar (Issue 27)
        pulse = Circle(radius=0.1, color=GOLD).move_to(bars[target_index].get_center())
        self.add(pulse)
        self.play(
            pulse.animate.scale(12).set_stroke(opacity=0),
            run_time=0.8,
            rate_func=rate_functions.ease_out_quad
        )
        self.remove(pulse)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE_CLR)
        self.lecture[2].set_color(GOLD)
        
        # 'Unchanged' label
        unchanged_label = Text("Unchanged", font_size=20, color=WHITE_CLR)
        self.place_at_grid(unchanged_label, "B3", scale_factor=1.0)

        non_target_indices = [i for i in range(len(bars)) if i != target_index]
        self.play(
            FadeIn(unchanged_label),
            *[Indicate(bars[i], color=WHITE_CLR, scale_factor=1.1) for i in non_target_indices],
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE_CLR)
        self.lecture[3].set_color(GOLD)
        
        # Scale negative gold bar slightly and label it '-A'
        minus_a_label = Text("-A", font_size=32, color=GOLD)
        # Position minus_a_label at the bottom of the inverted bar
        minus_a_label.next_to(bars[target_index], DOWN, buff=0.1)

        self.play(
            bars[target_index].animate.scale(1.1, about_edge=UP),
            Write(minus_a_label),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE_CLR)
        self.lecture[4].set_color(GOLD)
        
        # Final emphasis
        self.play(
            bars[target_index].animate.set_fill(opacity=1.0),
            Indicate(oracle_icon, color=GOLD),
            run_time=1
        )
        self.wait(2)
        
        # Cleanup
        self.lecture[4].set_color(WHITE_CLR)
        self.wait(1)
