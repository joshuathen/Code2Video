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
        title = "Real-World Application & Summary"
        lines = [
            "Hamming codes protect modern server memory.",
            "They prevent system crashes from cosmic rays.",
            "Self-healing data keeps our digital world reliable."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Load RAM stick asset (Issue 26)
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/ram.svg]
        ram_stick_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ram.svg", color="#00FFFF")
        ram_label = Text("ECC RAM", font_size=20, color="#00FFFF")
        ram_label.next_to(ram_stick_svg, UP, buff=0.1)
        
        ram_stick = VGroup(ram_stick_svg, ram_label)
        # Use scale factor 0.8 as per Issue 39
        self.place_in_area(ram_stick, "B2", "E5", scale_factor=0.8)
        
        self.play(FadeIn(ram_stick))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Cosmic Ray
        ray = Line(start=self.grid["A6"], end=ram_stick_svg.get_center(), color=WHITE, stroke_width=2)
        ray.add_tip(tip_length=0.15)
        
        # Bit within the RAM stick
        bit_1 = Text("1", font_size=24, color=GREEN).move_to(ram_stick_svg.get_center())
        bit_0 = Text("0", font_size=24, color=RED).move_to(ram_stick_svg.get_center())
        
        self.add(bit_1)
        self.play(Create(ray), run_time=1)
        
        # Bit flip (Error)
        self.play(ReplacementTransform(bit_1, bit_0), run_time=0.5)
        self.play(FadeOut(ray))
        
        # Error indicator
        error_box = SurroundingRectangle(bit_0, color=RED, buff=0.05)
        self.play(Create(error_box))
        self.wait(0.5)
        
        # Correction (Hamming magic)
        bit_1_new = Text("1", font_size=24, color=GREEN).move_to(ram_stick_svg.get_center())
        self.play(
            ReplacementTransform(bit_0, bit_1_new),
            error_box.animate.set_color(GREEN),
            run_time=1
        )
        self.play(FadeOut(error_box))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        summary_text = Text("Hamming Code: The Auto-Correct of Hardware", font_size=24, color=WHITE)
        # Fix placement as per Issue 38
        self.place_in_area(summary_text, 'F1', 'F6', scale_factor=0.8)
        
        self.play(Write(summary_text))
        self.wait(2)
