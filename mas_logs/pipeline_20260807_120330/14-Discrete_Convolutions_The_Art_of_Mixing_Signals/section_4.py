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
        # Setup the lecture lines and title
        self.setup_layout("Graphical Visualization", [
            "A stationary blue signal meets a sliding red kernel.",
            "Observe how the overlapping regions generate new values.",
            "Each step creates a point for the yellow output.",
            "Notice how convolution smooths out sharp signal edges."
        ])
        
        # Colors
        BLUE_PULSE = "#0000FF"
        RED_KERNEL = "#FF0000"
        YELLOW_OUTPUT = "#FFFF00"
        GREEN_HIGHLIGHT = "#00FF00"

        # Signal definitions
        pulse_indices = np.array([-1, 0, 1])
        pulse_vals = np.array([1, 1, 1])
        
        kernel_indices = np.array([-1, 0, 1])
        kernel_vals = np.array([0.5, 1, 0.5]) # Triangle kernel
        
        # Axes Setup
        # Top axes for sliding visualization (Issue 28)
        axes_top = Axes(
            x_range=[-5, 5, 1],
            y_range=[0, 1.5, 0.5],
            x_length=5,
            y_length=2.0,
            axis_config={"include_tip": False, "stroke_width": 1}
        ).set_color(GRAY)
        self.place_in_area(axes_top, "B1", "D6") # Fix: Moved from A1-D6 to B1-D6
        
        # Bottom axes for output visualization (Issue 29)
        axes_bottom = Axes(
            x_range=[-5, 5, 1],
            y_range=[0, 3, 1],
            x_length=5,
            y_length=0.8,
            axis_config={"include_tip": False, "stroke_width": 1}
        ).set_color(GRAY)
        self.place_in_area(axes_bottom, "F1", "F6") # Fix: Moved from E1-F6 to F1-F6

        # Labels (Issue 28, 29, 30)
        input_label = Text("Input f[n]", font_size=16, color=BLUE_PULSE)
        self.place_at_grid(input_label, "A2") # Fix: Center at A2
        
        kernel_label = Text("Kernel g[t-n]", font_size=16, color=RED_KERNEL)
        self.place_at_grid(kernel_label, "A4") # Fix: Center at A4
        
        output_label = Text("Output (f * g)[t]", font_size=16, color=YELLOW_OUTPUT)
        self.place_at_grid(output_label, "E3") # Fix: Center at E3

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE_PULSE)
        
        # Create stationary blue pulse bars
        pulse_bars = VGroup(*[
            Rectangle(width=0.3, height=val * 1.0, fill_color=BLUE_PULSE, fill_opacity=0.6, stroke_width=1)
            .move_to(axes_top.c2p(idx, val / 2))
            for idx, val in zip(pulse_indices, pulse_vals)
        ])
        
        self.play(Create(axes_top), FadeIn(pulse_bars), FadeIn(input_label))
        
        # Value tracker for the sliding shift 't'
        t_tracker = ValueTracker(-4)
        
        # Create sliding red kernel bars
        kernel_bars = VGroup(*[
            Rectangle(width=0.3, height=val * 1.0, fill_color=RED_KERNEL, fill_opacity=0.6, stroke_width=1)
            for val in kernel_vals
        ])
        
        def update_kernel(m):
            t = t_tracker.get_value()
            for i, bar in enumerate(m):
                # Center the kernel at shift t
                idx = kernel_indices[i]
                bar.move_to(axes_top.c2p(t + idx, kernel_vals[i] / 2))
        
        kernel_bars.add_updater(update_kernel)
        self.play(FadeIn(kernel_bars), FadeIn(kernel_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GREEN_HIGHLIGHT)
        
        # Green highlight bars represent the product f[n] * g[t-n]
        highlight_bars = VGroup(*[
            Rectangle(width=0.3, height=0.01, fill_color=GREEN_HIGHLIGHT, fill_opacity=0.8, stroke_width=1)
            .move_to(axes_top.c2p(idx, 0))
            for idx in pulse_indices
        ])
        highlight_bars.set_opacity(0)
        self.add(highlight_bars)
        
        def update_highlights(m):
            t = t_tracker.get_value()
            for i, bar in enumerate(m):
                n = pulse_indices[i]
                f_val = pulse_vals[i]
                # Calculate the value of the kernel g(t-n) at position n
                k_offset = n - t
                if -1 <= k_offset <= 0:
                    g_val = 0.5 + 0.5 * (k_offset + 1)
                elif 0 < k_offset <= 1:
                    g_val = 1 - 0.5 * k_offset
                else:
                    g_val = 0
                
                prod = f_val * g_val
                if prod > 0.05:
                    bar.stretch_to_fit_height(prod * 1.0)
                    bar.move_to(axes_top.c2p(n, prod / 2))
                    bar.set_opacity(0.8)
                else:
                    bar.set_opacity(0)
        
        highlight_bars.add_updater(update_highlights)
        
        # Slide kernel to show first overlap
        self.play(t_tracker.animate.set_value(-1), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW_OUTPUT)
        self.play(Create(axes_bottom), FadeIn(output_label))
        
        # Accumulating output bars at discrete steps
        output_steps = np.arange(-4, 4.1, 0.2)
        output_bars = VGroup()
        for step_x in output_steps:
            bar = Rectangle(width=0.1, height=0.01, fill_color=YELLOW_OUTPUT, fill_opacity=0.7, stroke_width=0)
            bar.move_to(axes_bottom.c2p(step_x, 0))
            bar.set_opacity(0)
            output_bars.add(bar)
        self.add(output_bars)
        
        def update_output(m):
            t = t_tracker.get_value()
            for i, bar in enumerate(m):
                step_x = output_steps[i]
                if step_x <= t:
                    # Discrete convolution value at step_x
                    y_val = 0
                    for j, f_val in enumerate(pulse_vals):
                        n = pulse_indices[j]
                        k_offset = n - step_x
                        if -1 <= k_offset <= 0:
                            g_val = 0.5 + 0.5 * (k_offset + 1)
                        elif 0 < k_offset <= 1:
                            g_val = 1 - 0.5 * k_offset
                        else:
                            g_val = 0
                        y_val += f_val * g_val
                    
                    if y_val > 0.05:
                        # Adjusted height scaling for shorter axes_bottom
                        bar.stretch_to_fit_height(y_val * 0.25)
                        bar.move_to(axes_bottom.c2p(step_x, y_val * 0.125))
                        bar.set_opacity(0.8)
        
        output_bars.add_updater(update_output)
        
        # Continuous slide to complete the convolution
        self.play(t_tracker.animate.set_value(4), run_time=6, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Final visual summary: smoothing effect
        self.lecture[3].set_color(YELLOW_OUTPUT)
        pulse_rect = SurroundingRectangle(pulse_bars, color=BLUE_PULSE, buff=0.1)
        output_rect = SurroundingRectangle(output_bars, color=YELLOW_OUTPUT, buff=0.1)
        
        self.play(Create(pulse_rect))
        self.wait(0.5)
        self.play(ReplacementTransform(pulse_rect, output_rect))
        self.wait(1)
        self.play(FadeOut(output_rect))
        
        self.wait(2)
