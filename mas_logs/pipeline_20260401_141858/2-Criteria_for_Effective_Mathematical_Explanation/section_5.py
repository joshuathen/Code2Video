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

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup title and lecture lines
        self.setup_layout(
            "Criterion 3: The Principle of Parsimony (Simplicity)", 
            [
                "Use the minimum steps required for rigor.", 
                "Avoid heavy jargon when plain language works.", 
                "Explain slope as 'up per step right.'"
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1
        self.play(self.lecture[0].animate.set_color("#A9A9A9"))
        
        # Cloud of jargon appearing in area A1-C6
        jargon1 = Text("differential quotient", font_size=18, color="#A9A9A9")
        jargon2 = Text("limit of the ratio", font_size=18, color="#A9A9A9")
        jargon3 = Text("infinitesimal change", font_size=18, color="#A9A9A9")
        
        self.place_at_grid(jargon1, "B2")
        self.place_at_grid(jargon2, "C4")
        self.place_at_grid(jargon3, "D3")
        
        jargon_group = VGroup(jargon1, jargon2, jargon3)
        self.play(FadeIn(jargon_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight lecture line 2
        self.play(self.lecture[1].animate.set_color("#FF0000"))
        
        # Large red 'X' strike-through
        x_line1 = Line(self.grid["A1"], self.grid["E6"], color="#FF0000", stroke_width=12)
        x_line2 = Line(self.grid["A6"], self.grid["E1"], color="#FF0000", stroke_width=12)
        
        self.play(Create(x_line1), Create(x_line2))
        self.wait(1)
        
        # Fade away jargon and X
        self.play(FadeOut(jargon_group), FadeOut(x_line1), FadeOut(x_line2))

        # === Animation for Lecture Line 3 ===
        # Highlight lecture line 3
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        
        # Simple staircase icon from assets
        staircase_asset = "/mmfs1/data/home/jthen/Code2Video/assets/icon/staircase.svg"
        staircase = SVGMobject(staircase_asset, color="#00FFFF")
        self.place_in_area(staircase, "B2", "E5", scale_factor=1.2)
        
        # Arrows for Up and Right
        up_arrow = Arrow(self.grid["E3"], self.grid["C3"], color="#00FFFF", buff=0.1)
        right_arrow = Arrow(self.grid["E3"], self.grid["E5"], color="#00FFFF", buff=0.1)
        
        # Labels
        up_text = Text("Up", font_size=20, color="#00FFFF")
        right_text = Text("Right", font_size=20, color="#00FFFF")
        self.place_at_grid(up_text, "D2")
        self.place_at_grid(right_text, "F4")
        
        slope_label = Text("Slope", font_size=32, color="#FFFFFF")
        self.place_at_grid(slope_label, "A3")
        
        # Animate simple explanation
        self.play(FadeIn(staircase), FadeIn(slope_label))
        self.play(GrowArrow(up_arrow), FadeIn(up_text))
        self.play(GrowArrow(right_arrow), FadeIn(right_text))
        self.wait(2)
