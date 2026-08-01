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

class Section1Scene(TeachingScene):
    def construct(self):
        # Initialize Scene
        title = "The Mystery of the Knowledge Base"
        lines = [
            "Where do Transformers store their massive encyclopedia of facts?",
            "Attention handles sequences, but MLPs hold the knowledge.",
            "Nearly two-thirds of parameters live within these MLP blocks."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_LINE1 = BLUE_B
        COLOR_LINE2 = "#FFD700"  # Gold
        COLOR_LINE3 = GREEN_B

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_LINE1))
        
        # Lexi the Robot [Asset Integration]
        lexi = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/robot.svg")
        lexi.set_color(COLOR_LINE1)
        self.place_at_grid(lexi, "C2", scale_factor=0.8)

        # Thought Bubble with Asset
        bubble_box = RoundedRectangle(corner_radius=0.2, height=1.2, width=3.2, color=WHITE)
        
        tower_asset = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/tower.svg")
        tower_asset.set_height(0.6).set_color(COLOR_LINE1)
        paris_text = Text(": Paris", font_size=18, color=COLOR_LINE1)
        content = VGroup(tower_asset, paris_text).arrange(RIGHT, buff=0.1)
        
        bubble_vgroup = VGroup(bubble_box, content)
        self.place_at_grid(bubble_vgroup, "B5", scale_factor=0.9)
        
        # Small circles for thought bubble connection
        c1 = Circle(radius=0.05, color=WHITE).move_to(self.grid["C3"])
        c2 = Circle(radius=0.1, color=WHITE).move_to(self.grid["C4"])

        self.play(Create(lexi))
        self.play(FadeIn(c1), FadeIn(c2), Create(bubble_vgroup))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_LINE2),
            FadeOut(lexi), FadeOut(bubble_vgroup), FadeOut(c1), FadeOut(c2)
        )

        # Transformer Block Diagram
        attn_box = Rectangle(height=0.8, width=3.5, color=WHITE, fill_opacity=0.1).set_fill(WHITE)
        attn_label = Text("Attention Mechanism", font_size=16, color=WHITE)
        attn_group = VGroup(attn_box, attn_label)
        
        mlp_box = Rectangle(height=1.2, width=3.5, color=COLOR_LINE2, fill_opacity=0.3).set_fill(COLOR_LINE2)
        mlp_label = Text("MLP Block", font_size=20, color=COLOR_LINE2, weight=BOLD)
        mlp_group = VGroup(mlp_box, mlp_label)

        arrow = Arrow(start=DOWN, end=UP, color=GRAY).scale(0.5)

        # Position diagram components in areas for better horizontal spread
        self.place_in_area(attn_group, 'D3', 'D5', scale_factor=0.9)
        self.place_in_area(mlp_group, 'B3', 'B5', scale_factor=0.9)
        arrow.next_to(attn_group, UP, buff=0.2)

        self.play(Create(attn_group))
        self.play(GrowArrow(arrow))
        self.play(Create(mlp_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_LINE3)
        )

        # Knowledge Stats - Adjusted scale for A4 positioning
        param_text = Text("66% of Parameters", font_size=24, color=COLOR_LINE3)
        self.place_at_grid(param_text, 'A4', scale_factor=0.8)
        
        highlight_box = mlp_box.copy().set_stroke(COLOR_LINE3, width=6).scale(1.1)

        self.play(
            Write(param_text),
            Create(highlight_box),
            mlp_box.animate.set_fill(COLOR_LINE3, opacity=0.5)
        )
        self.wait(2)
