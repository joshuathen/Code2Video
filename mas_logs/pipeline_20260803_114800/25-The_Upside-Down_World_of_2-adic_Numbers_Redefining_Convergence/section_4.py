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
        title_text = "The Paradoxical Sum: 1 + 2 + 4 + 8..."
        lecture_lines = [
            "In standard math, this sum diverges to infinity.",
            "In the 2-adic world, powers of two shrink.",
            "Higher powers approach zero in this new space.",
            "Large terms fit inside a finite 2-adic pouch.",
            "The infinite sum becomes a convergent series."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Display '1 + 2 + 4 + 8 +...' in #FF4500.
        self.lecture[0].set_color("#FF4500")
        sum_tex = MathTex("1 + 2 + 4 + 8 + \\dots", color="#FF4500")
        # Resolution of Issue 30: Move sum_tex to A4-A6
        self.place_in_area(sum_tex, "A4", "A6", scale_factor=0.8)
        
        inf_tex = MathTex("= \\infty", color="#FF4500")
        inf_tex.next_to(sum_tex, RIGHT)
        
        self.play(Write(sum_tex))
        self.play(Write(inf_tex))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A squirrel icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/squirrel.svg] appears holding a large '1' nut.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#DEB887")
        
        # Resolution of Issue 21: Integrated squirrel asset
        squirrel = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/squirrel.svg")
        squirrel.set_color("#DEB887").set_stroke(width=1)
        squirrel_label = Text("Squirrel", font_size=16, color=WHITE).next_to(squirrel, UP, buff=0.1)
        squirrel_group = VGroup(squirrel, squirrel_label)
        
        # Resolution of Issue 31: Move squirrel_group to D2
        self.place_at_grid(squirrel_group, "D2", scale_factor=0.8)
        
        nut1 = MathTex("1", color="#DEB887")
        # Resolution of Issue 31: Move nut1 to D3
        self.place_at_grid(nut1, "D3", scale_factor=1.0)
        
        self.play(FadeIn(squirrel_group))
        self.play(Create(nut1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Higher powers approach zero in this new space.
        # The squirrel picks up nut '2', then '4', then '8'.
        # Each nut '2^n' shrinks as the squirrel holds it.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#DEB887")
        
        nuts_holding_vgroup = VGroup(nut1)
        vals = [2, 4, 8, 16]
        
        # We'll spawn new nuts at various grid points then move them to the squirrel's "hand" (D3)
        spawn_points = ["B3", "C3", "E3", "F3"]
        
        for i, v in enumerate(vals):
            new_nut = MathTex(str(v), color="#DEB887")
            # Spawn at a grid location
            self.place_at_grid(new_nut, spawn_points[i], scale_factor=0.8)
            self.play(FadeIn(new_nut), run_time=0.5)
            
            # Shrink and move to the "holding" position (D3)
            shrink_factor = 0.8 / (1.8**(i+1))
            self.play(
                new_nut.animate.scale(shrink_factor/0.8).move_to(self.grid["D3"]),
                run_time=0.7
            )
            nuts_holding_vgroup.add(new_nut)
        
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Large terms fit inside a finite 2-adic pouch.
        # The squirrel drops all shrinking nuts into a tiny pouch.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#DEB887")
        
        pouch = RoundedRectangle(corner_radius=0.1, height=0.6, width=0.6, color=WHITE, fill_opacity=0.3)
        pouch_label = Text("Pouch", font_size=16, color=WHITE).next_to(pouch, DOWN, buff=0.1)
        pouch_group = VGroup(pouch, pouch_label)
        
        # Resolution of Issue 32: Move pouch_group to F3
        self.place_at_grid(pouch_group, "F3", scale_factor=1.0)
        
        self.play(FadeIn(pouch_group))
        
        # Move the squirrel and nuts towards the pouch, then "drop" the nuts in
        self.play(
            squirrel_group.animate.move_to(self.grid["E2"]),
            nuts_holding_vgroup.animate.move_to(self.grid["E3"]),
            run_time=1.0
        )
        
        # Nuts drop into the pouch and shrink even more
        self.play(
            nuts_holding_vgroup.animate.scale(0.3).move_to(self.grid["F3"]),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The infinite sum becomes a convergent series.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#FF4500")
        
        conv_label = Text("Convergent", font_size=24, color="#FF4500")
        conv_label.next_to(sum_tex, RIGHT, buff=0.2)
        
        # Transform the infinity symbol into "Convergent"
        self.play(
            Transform(inf_tex, conv_label)
        )
        self.wait(2)
