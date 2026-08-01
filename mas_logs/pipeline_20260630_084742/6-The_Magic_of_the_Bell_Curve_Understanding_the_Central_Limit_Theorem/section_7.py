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

class Section7Scene(TeachingScene):
    def construct(self):
        # Setup layout with specific lines
        lecture_lines = [
            'We moved from chaotic squirrels to perfect order.',
            'The Central Limit Theorem is the foundation of statistics.',
            'Chaos becomes predictable through the power of averages.'
        ]
        self.setup_layout("Summary & Conclusion", lecture_lines)

        # Initialize all lecture lines to a dimmed color
        for line in self.lecture:
            line.set_color(GRAY)

        # === Animation for Lecture Line 1 ===
        # Goal: Display collage (Chaos, Machine, Bell Curve)
        self.play(self.lecture[0].animate.set_color(WHITE))

        # 1. Chaos: A cluster of random dots
        chaos_group = VGroup(*[
            Dot(point=[np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4), 0], 
                radius=0.04, color="#FF6347") 
            for _ in range(25)
        ])
        self.place_in_area(chaos_group, 'B1', 'C2', scale_factor=1.0)
        chaos_label = Text("Chaos", font_size=16, color="#FF6347")
        self.place_at_grid(chaos_label, 'D1', scale_factor=1.0)

        # 2. Sampling Machine: Simple icon
        machine_body = Rectangle(width=1.0, height=0.8, color="#4682B4", fill_opacity=0.2)
        machine_gear = Star(n=8, outer_radius=0.25, inner_radius=0.15, color="#4682B4")
        machine = VGroup(machine_body, machine_gear)
        self.place_in_area(machine, 'B3', 'C4', scale_factor=1.0)
        machine_label = Text("Sampling", font_size=16, color="#4682B4")
        # Fixed Issue 51: Center machine_label under the machine area
        self.place_in_area(machine_label, 'D3', 'D4', scale_factor=1.0)

        # 3. Bell Curve: Standard Gaussian
        bell_axes = Axes(x_range=[-3, 3], y_range=[0, 1], axis_config={"include_tip": False}).set_opacity(0)
        bell_curve_small = bell_axes.plot(lambda x: np.exp(-x**2), color="#FFD700")
        bell_group = VGroup(bell_curve_small)
        self.place_in_area(bell_group, 'B5', 'C6', scale_factor=1.0)
        bell_label = Text("Order", font_size=16, color="#FFD700")
        self.place_at_grid(bell_label, 'D5', scale_factor=1.0)

        collage = VGroup(chaos_group, chaos_label, machine, machine_label, bell_group, bell_label)
        self.play(FadeIn(collage))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Goal: Scale the Bell Curve to fill the area, others fade
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color("#FFD700")
        )

        # Create a larger bell curve for transition
        large_bell_axes = Axes(x_range=[-3, 3], y_range=[0, 1.2], axis_config={"include_tip": False}).set_opacity(0)
        large_bell_curve = large_bell_axes.plot(lambda x: np.exp(-x**2), color="#FFD700", stroke_width=6)
        large_bell_group = VGroup(large_bell_curve)
        # Fixed Issue 50: Set scale factor to 1.0 to prevent obstruction
        self.place_in_area(large_bell_group, 'A1', 'E6', scale_factor=1.0)

        self.play(
            FadeOut(chaos_group), FadeOut(chaos_label),
            FadeOut(machine), FadeOut(machine_label),
            FadeOut(bell_label),
            ReplacementTransform(bell_group, large_bell_group),
            run_time=1.5
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Goal: Fade in final text
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color(WHITE)
        )

        final_msg = Text("Order from chaos through the CLT", font_size=28, color=WHITE)
        # Fixed Issue 52: Reduced scale factor to avoid tight margins
        self.place_in_area(final_msg, 'F1', 'F6', scale_factor=0.8)

        self.play(Write(final_msg))
        self.wait(3)
