from manim import *

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
        lecture_lines = [
            'Total time T equals distance divided by speed.',
            'To find the minimum time, set dT/dx to zero.',
            'The derivative reveals ratios of distances and speeds.',
            'Geometry shows these ratios are sines of the angles.',
            "This optimization leads directly to light's final path."
        ]
        self.setup_layout("The Mathematical Derivation", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Using Text instead of MathTex to avoid FileNotFoundError: 'latex'
        t_func_vgroup = VGroup(
            Text("T(x) =", font_size=24),
            Text("√(a² + x²) / v₁", font_size=24),
            Text("+", font_size=24),
            Text("√(b² + (L-x)²) / v₂", font_size=24)
        ).arrange(RIGHT, buff=0.15)
        
        self.place_in_area(t_func_vgroup, "A1", "A6", scale_factor=0.85)
        
        brace1 = Brace(t_func_vgroup[1], DOWN, buff=0.1)
        l1_label = Text("L₁ / v₁", font_size=18).next_to(brace1, DOWN, buff=0.1)
        brace2 = Brace(t_func_vgroup[3], DOWN, buff=0.1)
        l2_label = Text("L₂ / v₂", font_size=18).next_to(brace2, DOWN, buff=0.1)
        
        self.play(Write(t_func_vgroup))
        self.play(Create(brace1), FadeIn(l1_label), Create(brace2), FadeIn(l2_label))
        self.wait(1.5)
        
        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        deriv_op = Text("d/dx [ T(x) ] = 0", color=YELLOW, font_size=28)
        self.place_at_grid(deriv_op, "B3")
        
        self.play(Write(deriv_op))
        self.wait(1.5)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        calc_step = Text("x / (v₁ √(a²+x²)) = (L-x) / (v₂ √(b²+(L-x)²))", font_size=20)
        # Resolved Issue 41: Adjusted scale factor to avoid crowding
        self.place_in_area(calc_step, 'C1', 'C6', scale_factor=0.75)
        
        self.play(Write(calc_step))
        self.wait(1.5)
        
        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        sine_step = Text("sin(θ₁) / v₁ = sin(θ₂) / v₂", font_size=24)
        # Resolved Issue 42: Center across full width for better alignment
        self.place_in_area(sine_step, 'D1', 'D6', scale_factor=0.9)
        
        self.play(Write(sine_step))
        # Visual pulse for the ratios
        self.play(sine_step.animate.set_color(YELLOW), run_time=0.5)
        self.play(sine_step.animate.set_color(WHITE), run_time=0.5)
        self.wait(1.5)
        
        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        snells_law = Text("n₁ sin(θ₁) = n₂ sin(θ₂)", font_size=32, color=GOLD)
        # Resolved Issue 43: Final result highlighted and centered
        self.place_in_area(snells_law, 'E1', 'E6', scale_factor=1.1)
        
        self.play(Write(snells_law))
        self.play(Indicate(snells_law))
        self.wait(2)
