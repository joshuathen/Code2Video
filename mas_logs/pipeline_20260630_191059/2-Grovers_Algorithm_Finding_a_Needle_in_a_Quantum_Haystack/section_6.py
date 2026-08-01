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

class Section6Scene(TeachingScene):
    def construct(self):
        # Data from storyboard
        title = "Quantum Speedup & Conclusion"
        lines = [
            "We only need roughly square root of N iterations.",
            "This provides a massive quadratic speedup over classical methods.",
            "Measuring the system now reveals the correct answer."
        ]
        
        self.setup_layout(title, lines)
        
        # === Animation for Lecture Line 1 ===
        # Show a comparison table of 'Classical: 1,000,000' vs 'Quantum: 1,000' in #FFFFFF.
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        classical_text = Text("Classical: 1,000,000", font_size=24, color=WHITE)
        quantum_text = Text("Quantum: 1,000", font_size=24, color=WHITE)
        comparison = VGroup(classical_text, quantum_text).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        
        # Issue 33: Move to A2-B5 and scale 0.9 to improve visual balance.
        self.place_in_area(comparison, "A2", "B5", scale_factor=0.9)
        
        self.play(FadeIn(comparison))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Display a green progress bar filling to '31 steps' to represent O(sqrt(N)) speedup.
        self.play(self.lecture[1].animate.set_color(GREEN))
        
        bar_width = 3.5
        bar_bg = Rectangle(width=bar_width, height=0.4, color=WHITE, stroke_width=2)
        bar_fill = Rectangle(width=0.01, height=0.4, color=GREEN, fill_opacity=1, stroke_width=0)
        bar_fill.align_to(bar_bg, LEFT)
        
        bar_label = Text("31 steps", font_size=22, color=GREEN)
        sqrt_n_label = Text("O(√N)", font_size=28, color=GREEN)
        
        # Issue 32: Group elements and scale to 0.8 to prevent vertical compression in C1-D6.
        bar_rects = VGroup(bar_bg, bar_fill)
        sqrt_n_label.next_to(bar_rects, UP, buff=0.3)
        bar_label.next_to(bar_rects, DOWN, buff=0.3)
        
        full_bar_group = VGroup(sqrt_n_label, bar_rects, bar_label)
        self.place_in_area(full_bar_group, "C1", "D6", scale_factor=0.8)
        
        self.play(
            Create(bar_bg),
            Write(sqrt_n_label)
        )
        self.play(
            bar_fill.animate.stretch_to_fit_width(bar_bg.get_width(), about_edge=LEFT),
            FadeIn(bar_label),
            run_time=2.0
        )
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Fade in a solid Golden Key icon in #FFD700 to represent the final measurement.
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        
        # Golden Key representation using basic shapes
        key_handle = Circle(radius=0.25, color="#FFD700", stroke_width=4)
        key_shaft = Rectangle(width=0.6, height=0.1, color="#FFD700", fill_opacity=1, stroke_width=0)
        key_shaft.next_to(key_handle, RIGHT, buff=0)
        notch1 = Rectangle(width=0.08, height=0.15, color="#FFD700", fill_opacity=1, stroke_width=0)
        notch1.move_to(key_shaft.get_right() + LEFT*0.1 + DOWN*0.1)
        notch2 = Rectangle(width=0.08, height=0.15, color="#FFD700", fill_opacity=1, stroke_width=0)
        notch2.move_to(key_shaft.get_right() + LEFT*0.25 + DOWN*0.1)
        
        golden_key = VGroup(key_handle, key_shaft, notch1, notch2)
        
        # Issue 31: Reduce scale_factor to 1.1 to avoid crowding labels and bar.
        self.place_in_area(golden_key, "E2", "F5", scale_factor=1.1)
        
        self.play(FadeIn(golden_key, scale=1.1))
        self.play(Flash(golden_key, color="#FFD700", line_length=0.3, num_lines=12))
        self.wait(3)
