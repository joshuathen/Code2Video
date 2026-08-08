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

class Section2Scene(TeachingScene):
    def construct(self):
        # Initialize Scene data from shared state
        title = "Prerequisite: The 2-adic Valuation"
        lines = [
            "In 2-adic distance, high powers of 2 are small.",
            "A sieve shows 64 is closer to 0 than 2.",
            "Multiples of 1024 hit the target's bullseye."
        ]
        self.setup_layout(title, lines)

        # Colors as defined in the storyboard
        COLOR_RINGS = "#444444"
        COLOR_ZERO = "#FFFFFF"
        COLOR_NUMS = "#00FF00"
        COLOR_ARROWS = "#8888FF"
        COLOR_LABELS = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # "In 2-adic distance, high powers of 2 are small."
        self.lecture[0].set_color(COLOR_NUMS)
        
        # Load and place target asset as requested in Issue 16
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/target.svg]
        target_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/target.svg")
        self.place_at_grid(target_asset, "D3", scale_factor=1.2)
        target_asset.set_color(COLOR_RINGS)
        
        # Draw target with 5 concentric gray rings
        # Center the target at grid D3
        rings = VGroup(*[
            Circle(radius=r, color=COLOR_RINGS, stroke_width=2) 
            for r in [0.5, 1.0, 1.5, 2.0, 2.5]
        ])
        self.place_at_grid(rings, 'D3')
        
        zero_label = Text("0", color=COLOR_ZERO)
        self.place_at_grid(zero_label, 'D3', scale_factor=0.6)
        
        self.play(DrawBorderThenFill(target_asset), Create(rings), Write(zero_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "A sieve shows 64 is closer to 0 than 2."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_LABELS)
        
        # Place numbers '2', '4', '16', and '64' (#00FF00)
        num2 = Text("2", color=COLOR_NUMS)
        num4 = Text("4", color=COLOR_NUMS)
        num16 = Text("16", color=COLOR_NUMS)
        num64 = Text("64", color=COLOR_NUMS)
        
        # Position mapping relative to D3
        # num2 and num4 are in the outer rings (dist 2-3)
        # num16 and num64 are in the inner rings (dist 1)
        self.place_at_grid(num2, 'B3', scale_factor=0.7)    # 2.0 units from D3 (Row B is safe per B005)
        self.place_at_grid(num4, 'F3', scale_factor=0.6)    # 2.0 units from D3
        self.place_at_grid(num16, 'E3', scale_factor=0.5)   # 1.0 unit from D3
        self.place_at_grid(num64, 'D4', scale_factor=0.4)   # 1.0 unit from D3 (Fixed per Issue 22)
        
        self.play(Write(num2), Write(num4), Write(num16), Write(num64))
        self.wait(1)
        
        # Animate blue arrows (#8888FF) and yellow labels (#FFFF00)
        # Arrow from center (D3) to '2' (B3)
        arrow_2 = Arrow(start=self.grid['D3'], end=self.grid['B3'], color=COLOR_ARROWS, buff=0.1)
        label_2 = MathTex(r"|2|_2 = 1/2", color=COLOR_LABELS)
        # Place label spanning B4-B5 to accommodate width and symmetry
        self.place_in_area(label_2, 'B4', 'B5', scale_factor=0.7) 
        
        self.play(Create(arrow_2), Write(label_2))
        self.wait(1)
        
        # Arrow from center (D3) to '64' (D4)
        arrow_64 = Arrow(start=self.grid['D3'], end=self.grid['D4'], color=COLOR_ARROWS, buff=0.1)
        label_64 = MathTex(r"|64|_2 = 1/64", color=COLOR_LABELS)
        # Place label at E4-E5 (Fixed per Issue 21)
        self.place_in_area(label_64, 'E4', 'E5', scale_factor=0.7)
        
        self.play(Create(arrow_64), Write(label_64))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # "Multiples of 1024 hit the target's bullseye."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_ZERO)
        
        # Pulse the center '0' to show it is the limit for large powers of 2
        self.play(
            zero_label.animate.scale(1.5).set_color(YELLOW),
            run_time=0.5
        )
        self.play(
            zero_label.animate.scale(1/1.5).set_color(COLOR_ZERO),
            run_time=0.5
        )
        self.play(Flash(self.grid['D3'], color=COLOR_ZERO, line_length=0.3))
        
        self.wait(2)
