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
        # 1. Setup Layout
        title_str = "The Change of Basis Matrix (The Translator)"
        lecture_lines = [
            "Express the new basis vectors in standard coordinates.",
            "Arrange these vectors as columns in matrix P.",
            "Multiplying by P converts coordinates back to standard."
        ]
        self.setup_layout(title_str, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight current lecture line
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        
        # Matrix P symbol labeled "The Translator" (Green: #00FF00)
        p_symbol = Text("P", color="#00FF00", font_size=80)
        self.place_at_grid(p_symbol, "B3")
        
        # [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/translator.svg]
        translator_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/translator.svg")
        translator_icon.set_color("#00FF00")
        self.place_at_grid(translator_icon, "B5", scale_factor=0.6)
        
        # Multi-word label positioning (Visual Anchor System)
        translator_label = Text("The Translator", color="#00FF00", font_size=28)
        self.place_in_area(translator_label, "C2", "C5", scale_factor=1.0)
        
        self.play(
            FadeIn(p_symbol),
            FadeIn(translator_icon),
            Write(translator_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight next lecture line
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFD700")
        )
        
        # Expand P using Text to avoid LaTeX dependency (FileNotFoundError: latex)
        p_expanded = VGroup(
            Text("P=[ b1  b2 ]", color="#FFD700", font_size=42)
        )
        self.place_in_area(p_expanded, "D2", "D5", scale_factor=1.0)
        
        self.play(
            ReplacementTransform(p_symbol, p_expanded[0][0]),
            Write(p_expanded[0][1:]),
            FadeOut(translator_icon),
            FadeOut(translator_label)
        )
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Highlight final lecture line
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#87CEEB")
        )
        
        # Matrix-vector product P * [x]_New = [x]_Std using VGroup of Text
        # Color coding: P (Gold), [x]_New (Gold), = (White), [x]_Std (Sky Blue)
        formula = VGroup(
            Text("P"), 
            Text("[x]New"), 
            Text("="), 
            Text("[x]Std")
        ).arrange(RIGHT, buff=0.2)
        
        for mob in formula:
            mob.set_font_size(48)
            
        formula[0].set_color("#FFD700")
        formula[1].set_color("#FFD700")
        formula[2].set_color(WHITE)
        formula[3].set_color("#87CEEB")
        
        self.place_in_area(formula, "F2", "F5", scale_factor=1.0)
        
        self.play(Write(formula))
        self.wait(1)
        
        # Highlight complete formula with white flash (#FFFFFF)
        self.play(Flash(formula, color="#FFFFFF", flash_radius=1.5, line_length=0.4))
        self.play(formula.animate.set_stroke(width=1, color=WHITE))
        self.play(formula.animate.set_stroke(width=0))
        
        self.wait(2)
