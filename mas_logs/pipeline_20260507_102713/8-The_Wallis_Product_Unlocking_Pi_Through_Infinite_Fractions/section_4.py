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
        # Initialize Layout
        self.setup_layout(
            "The Ratio and the Sandwich Theorem",
            [
                "Higher powers of sine always produce smaller integral areas.",
                "The ratio of consecutive integrals approaches one at infinity.",
                "This squeeze theorem forces the two paths to meet."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Higher powers of sine always produce smaller integral areas.
        # FIX: Replaced MathTex with Text to avoid FileNotFoundError: 'latex' 
        # when the system environment lacks a LaTeX distribution.
        ineq1 = Text("I_{n+1} < I_n < I_{n-1}", color=WHITE, font_size=24)
        # Using place_in_area to ensure centering and avoid lecture overlap
        self.place_in_area(ineq1, "B1", "B6", scale_factor=1.1)
        
        self.play(
            Write(ineq1),
            self.lecture[0].animate.set_color(WHITE),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The ratio of consecutive integrals approaches one at infinity.
        # FIX: Replaced MathTex with Text and simplified notation for non-LaTeX rendering.
        ineq2 = Text(
            "I_{n+1} / I_{n-1} < I_n / I_{n-1} < 1",
            color="#ADD8E6",
            font_size=22
        )
        # Positioned lower in the grid to avoid crowding
        self.place_in_area(ineq2, "D1", "D6", scale_factor=1.0)
        
        self.play(
            ReplacementTransform(ineq1.copy(), ineq2),
            self.lecture[1].animate.set_color("#ADD8E6"),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # This squeeze theorem forces the two paths to meet.
        # FIX: Replaced MathTex with Text and used standard characters.
        
        limit_expr = Text("lim (n->inf) I_n / I_{n-1} = 1", color=YELLOW, font_size=22)
        self.place_at_grid(limit_expr, "F2", scale_factor=1.0)
        
        # Load and place the Asset
        try:
            sandwich_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/sandwich.svg")
            self.place_at_grid(sandwich_icon, "F5", scale_factor=0.8)
            asset_anim = FadeIn(sandwich_icon)
        except:
            # Fallback for local dev if asset is missing
            sandwich_icon = Square(color=YELLOW).scale(0.5)
            self.place_at_grid(sandwich_icon, "F5", scale_factor=0.8)
            asset_anim = Create(sandwich_icon)

        self.play(
            Write(limit_expr),
            asset_anim,
            self.lecture[2].animate.set_color(YELLOW),
            run_time=2
        )
        self.wait(3)
