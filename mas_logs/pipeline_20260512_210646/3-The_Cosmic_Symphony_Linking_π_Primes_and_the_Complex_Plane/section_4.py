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
        lecture_lines = [
            'Euler bridged the gap between integers and prime numbers.',
            'Every integer sum transforms into a product of primes.',
            'This equality is known as the Euler Product formula.',
            'It proves that primes govern the behavior of circles.',
            'This connects prime distribution directly to the zeta function.'
        ]
        self.setup_layout("The Euler Product: Primes Join the Party", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        
        # Zeta sum formula on the left (#FFFFFF) - Issue 37 fix
        zeta_sum = Text("Σ (1 / nˢ)", font_size=32, color="#FFFFFF")
        self.place_in_area(zeta_sum, "C1", "C2", scale_factor=0.8)
        
        # Euler product formula on the right (#00FFFF) - Issue 36 fix
        euler_prod = Text("Π (1 / (1 - p⁻ˢ))", font_size=32, color="#00FFFF")
        self.place_in_area(euler_prod, "C5", "C6", scale_factor=0.8)
        
        self.play(FadeIn(zeta_sum), FadeIn(euler_prod))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        
        # Primes moving from product side to sum side - Issue 35 fix
        primes_list = ["2", "3", "5", "7", "11"]
        prime_mobs = VGroup(*[Text(p, font_size=28, color="#00FFFF") for p in primes_list])
        
        for mob in prime_mobs:
            # Place at B5 instead of C5 to avoid formula overlap
            self.place_at_grid(mob, "B5", scale_factor=0.7)
            
        self.play(
            LaggedStart(
                *[mob.animate.move_to(self.grid["C2"]).set_opacity(0) for mob in prime_mobs],
                lag_ratio=0.4,
                run_time=2.5
            )
        )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        
        # Large equals sign in flashing yellow (#FFFF00)
        equals_sign = Text("=", font_size=64, color="#FFFF00")
        self.place_in_area(equals_sign, "C3", "C4")
        
        self.play(Write(equals_sign))
        self.play(Indicate(equals_sign, color="#FFFF00", scale_factor=1.3), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(WHITE)
        
        # Highlight term π²/6 from previous section
        pi_val = Text("π² / 6", font_size=32, color="#FFFFFF")
        self.place_at_grid(pi_val, "E2")
        
        s_label = Text("(for s = 2)", font_size=20, color="#FFFFFF")
        self.place_at_grid(s_label, "D2")
        
        link_arrow = Arrow(self.grid["E2"], self.grid["C2"], color="#FFFF00", buff=0.1)
        
        self.play(FadeIn(pi_val), FadeIn(s_label), Create(link_arrow))
        self.play(Indicate(pi_val, color="#FFFF00"))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FF4500")
        
        # Single glowing 'Zeta' symbol (#FF4500)
        zeta_symbol = Text("ζ(s)", font_size=84, color="#FF4500")
        self.place_in_area(zeta_symbol, "C3", "C4")
        
        # Cleanup Line 4 indicators
        self.play(FadeOut(pi_val), FadeOut(s_label), FadeOut(link_arrow))
        
        # Transform formulas into the single glowing Zeta
        self.play(
            ReplacementTransform(VGroup(zeta_sum, euler_prod, equals_sign), zeta_symbol),
            run_time=2
        )
        
        # Glowing effect pulse
        self.play(
            zeta_symbol.animate.scale(1.1),
            rate_func=there_and_back,
            run_time=1.5
        )
        self.wait(2)
