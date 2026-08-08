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
        # Initialize layout
        self.setup_layout("Conclusion: The Hidden Harmony", [
            "This experiment acts as a hidden geometric calculator.",
            "Pi is fundamental to motion and energy conservation.",
            "Mathematics connects the simple \"clack\" to the universe."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Fade in a large glowing gold Pi symbol (#FFD700).
        pi_sym = MathTex(r"\pi", color="#FFD700", font_size=144)
        pi_glow = MathTex(r"\pi", color="#FFD700", font_size=150).set_opacity(0.3)
        pi_group = VGroup(pi_glow, pi_sym)
        
        # Fix for Issue 34: Scaling down to 0.9 to prevent dominating the grid
        self.place_in_area(pi_group, "B3", "D4", scale_factor=0.9)
        
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        self.play(FadeIn(pi_group, shift=UP*0.2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Flash labels 'Momentum' and 'Energy' around the Pi symbol.
        mom_lab = Text("Momentum", color="#00FF00", font_size=24)
        ene_lab = Text("Energy", color="#00FF00", font_size=24)
        
        # Fix for Issue 35: Moving mom_lab to B2 and scaling to 0.8
        self.place_at_grid(mom_lab, "B2", scale_factor=0.8)
        # Fix for Issue 36: Moving ene_lab to B5 and scaling to 0.8
        self.place_at_grid(ene_lab, "B5", scale_factor=0.8)
        
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        # Indication provides the "flash" effect requested in the plan
        self.play(
            FadeIn(mom_lab),
            FadeIn(ene_lab),
            Indicate(mom_lab, color="#00FF00"),
            Indicate(ene_lab, color="#00FF00")
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Final flash on the counter showing 3.1415926... in #FFFFFF.
        counter_val = Text("3.1415926...", color="#FFFFFF", font_size=32)
        # Positioned below the Pi symbol to avoid overlapping with Row A/F (B005)
        # pi_group spans B3 to D4, counter at E3-E4 is safe (B001)
        self.place_in_area(counter_val, "E3", "E4")
        
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        self.play(
            FadeIn(counter_val),
            Indicate(counter_val, scale_factor=1.2, color=WHITE)
        )
        self.wait(2)
