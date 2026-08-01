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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            'Composite functions are like one gear driving another.', 
            'The outer rate multiplies the inner rate of change.', 
            'Think of nested dolls: outside first, then the inside.', 
            'Differentiate the outer shell, keeping the inner part intact.', 
            'Then, multiply by the derivative of the inner function.'
        ]
        self.setup_layout("The Chain Rule: Linked Gears", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Three linked gears [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/gears.svg]
        # (A #FFD700, B #FFA500, C #FF4500) appear to represent nested functions.
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        
        try:
            # Asset integration for Issue 21
            gears_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/gears.svg")
            if len(gears_svg) >= 3:
                gear_a = gears_svg[0].set_color("#FFD700")
                gear_b = gears_svg[1].set_color("#FFA500")
                gear_c = gears_svg[2].set_color("#FF4500")
                gears = VGroup(gear_a, gear_b, gear_c).arrange(RIGHT, buff=0.2)
            else:
                gears = gears_svg.set_color("#FFD700")
        except:
            # Fallback to simple gears if asset is missing
            gear_a = Star(n=12, inner_radius=0.3, outer_radius=0.5, color="#FFD700", fill_opacity=1)
            gear_b = Star(n=12, inner_radius=0.25, outer_radius=0.4, color="#FFA500", fill_opacity=1)
            gear_c = Star(n=12, inner_radius=0.2, outer_radius=0.3, color="#FF4500", fill_opacity=1)
            gears = VGroup(gear_a, gear_b, gear_c).arrange(RIGHT, buff=0.1)

        self.place_in_area(gears, "B2", "D5", scale_factor=1.2)
        self.play(FadeIn(gears))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Gear A turns, causing Gear B and then Gear C to turn.
        # Ratios dy/du and du/dx appear.
        self.play(self.lecture[1].animate.set_color("#FFA500"))
        
        ratio1 = Text("dy/du", font_size=24, color="#FFA500")
        ratio2 = Text("du/dx", font_size=24, color="#FF4500")
        
        # Position ratios above gear gaps
        self.place_at_grid(ratio1, "B3", scale_factor=0.8)
        self.place_at_grid(ratio2, "B5", scale_factor=0.8)
        
        def rotate_logic(mob, dt):
            if isinstance(mob, VGroup) and len(mob) >= 3:
                mob[0].rotate(dt * 1.5)
                mob[1].rotate(-dt * 2.0)
                mob[2].rotate(dt * 3.0)
            else:
                mob.rotate(dt * 2)

        gears.add_updater(rotate_logic)
        self.play(Write(ratio1), Write(ratio2))
        self.wait(2)
        gears.remove_updater(rotate_logic)

        # === Animation for Lecture Line 3 ===
        # Think of nested dolls: outside first, then the inside.
        # Nested Doll visual: f(u) outer, g(x) inner.
        self.play(self.lecture[2].animate.set_color("#FF4500"))
        
        # Transition from gears to dolls
        doll_outer = Circle(radius=0.9, color="#FFD700", fill_opacity=0.2)
        doll_inner = Circle(radius=0.5, color="#FF4500", fill_opacity=0.4)
        label_f = Text("f(u)", font_size=24, color="#FFD700")
        label_g = Text("g(x)", font_size=24, color="#FF4500")
        
        doll_system = VGroup(doll_outer, doll_inner, label_f, label_g)
        self.place_in_area(doll_system, "B2", "D5", scale_factor=1.1)
        # Move labels relative to circles
        label_f.next_to(doll_outer, UP, buff=0.1)
        label_g.move_to(doll_inner.get_center())

        self.play(FadeOut(gears), FadeOut(ratio1), FadeOut(ratio2))
        self.play(Create(doll_outer), Write(label_f))
        self.wait(0.5)
        self.play(Create(doll_inner), Write(label_g))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Differentiate the outer shell, keeping the inner part intact.
        self.play(self.lecture[3].animate.set_color("#FFFFFF"))
        
        # Issue 30 logic: spacing elements out
        outer_diff = Text("f'(g(x))", font_size=32, color="#FFD700")
        self.place_at_grid(outer_diff, "E2", scale_factor=1.0)
        
        self.play(Indicate(doll_outer))
        self.play(Write(outer_diff))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Multiply by the derivative of the inner function.
        # Formula dy/dx = (dy/du) * (du/dx) fades in center.
        self.play(self.lecture[4].animate.set_color("#FFFFFF"))
        
        # Issue 30 logic: spacing elements out
        inner_diff = Text("⋅ g'(x)", font_size=32, color="#FF4500")
        self.place_at_grid(inner_diff, "E4", scale_factor=1.0)
        
        # Issue 28/29 logic: long formula in area F1-F6
        formula = Text("dy/dx = (dy/du) ⋅ (du/dx)", font_size=34, color=WHITE)
        self.place_in_area(formula, "F1", "F6", scale_factor=0.9)
        
        self.play(Indicate(doll_inner))
        self.play(Write(inner_diff))
        self.wait(1)
        self.play(FadeIn(formula))
        self.play(Indicate(formula))
        self.wait(2)
