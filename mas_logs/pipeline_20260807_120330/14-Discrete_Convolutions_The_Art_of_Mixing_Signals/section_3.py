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
        lecture_lines = [
            "First, flip the kernel to reverse its order.",
            "Second, shift the flipped kernel to your target index.",
            "Third, multiply overlapping values from the signal and kernel.",
            "Finally, sum these products for a single output point."
        ]
        self.setup_layout("The Four-Step Algorithm", lecture_lines)
        
        # Colors defined in requirements
        COLOR_SIGNAL = BLUE
        COLOR_KERNEL = "#FF0000"  # Red
        COLOR_OVERLAP = "#00FF00" # Green
        COLOR_SUM = "#FFFF00"     # Yellow

        # Signal f[k] - Moved to Row A to utilize space (Resolves Issue 25)
        signal_vals = [0.5, 1.2, 0.8]
        signal_bars = VGroup(*[
            Rectangle(width=0.6, height=val, fill_opacity=0.8, fill_color=COLOR_SIGNAL, stroke_color=WHITE)
            for val in signal_vals
        ])
        for i, bar in enumerate(signal_bars):
            self.place_at_grid(bar, f"A{i+2}")
        
        f_label = MathTex("f[k]", font_size=24, color=COLOR_SIGNAL)
        self.place_at_grid(f_label, "A1") 
        
        # Kernel g[k] - Moved to Row B (Resolves Issue 26)
        kernel_vals = [0.3, 0.6, 0.9]
        kernel_bars = VGroup(*[
            Rectangle(width=0.6, height=val, fill_opacity=0.8, fill_color=COLOR_KERNEL, stroke_color=WHITE)
            for val in kernel_vals
        ])
        # Initially offset to the right (B3-B5) to demonstrate shifting
        for i, bar in enumerate(kernel_bars):
            self.place_at_grid(bar, f"B{i+3}")
            
        g_label = MathTex("g[k]", font_size=24, color=COLOR_KERNEL)
        self.place_at_grid(g_label, "B1") 
        
        self.add(signal_bars, f_label, kernel_bars, g_label)
        self.wait(1)

        # === Animation for Lecture Line 1 ===
        # First, flip the kernel to reverse its order.
        self.play(self.lecture[0].animate.set_color(COLOR_KERNEL))
        
        # Reverse the order of heights visually
        target_heights = [bar.height for bar in reversed(kernel_bars)]
        flip_anims = [
            bar.animate.stretch_to_fit_height(h, about_edge=DOWN)
            for bar, h in zip(kernel_bars, target_heights)
        ]
            
        self.play(*flip_anims, run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Second, shift the flipped kernel to your target index.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_KERNEL)
        )
        
        # Slide kernel bars horizontally to align with signal (move to B2-B4)
        shift_anims = [
            bar.animate.move_to(self.grid[f"B{i+2}"])
            for i, bar in enumerate(kernel_bars)
        ]
            
        self.play(*shift_anims, run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Third, multiply overlapping values from the signal and kernel.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_OVERLAP)
        )
        
        # Pulse overlapping areas
        overlap_group = VGroup(*signal_bars, *kernel_bars)
        self.play(
            overlap_group.animate.set_fill(COLOR_OVERLAP),
            rate_func=there_and_back,
            run_time=1.5
        )
        
        # Show multiplication results as floating numbers in Row C
        # Heights: Signal (0.5, 1.2, 0.8) * Flipped Kernel (0.9, 0.6, 0.3)
        products = [0.45, 0.72, 0.24] 
        prod_labels = VGroup(*[
            MathTex(f"{p:.2f}", font_size=20, color=COLOR_OVERLAP)
            for p in products
        ])
        for i, label in enumerate(prod_labels):
            self.place_at_grid(label, f"C{i+2}")
            
        self.play(FadeIn(prod_labels))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Finally, sum these products for a single output point.
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(COLOR_SUM)
        )
        
        sum_val = sum(products) # 1.41
        sum_bar = Rectangle(width=0.6, height=sum_val, fill_opacity=0.8, fill_color=COLOR_SUM, stroke_color=WHITE)
        self.place_at_grid(sum_bar, "E3")
        
        y_label = MathTex("y[n]", font_size=24, color=COLOR_SUM)
        self.place_at_grid(y_label, "E2") # Resolved Issue 27
        
        self.play(
            FadeIn(y_label),
            ReplacementTransform(prod_labels, sum_bar)
        )
        self.wait(2)
