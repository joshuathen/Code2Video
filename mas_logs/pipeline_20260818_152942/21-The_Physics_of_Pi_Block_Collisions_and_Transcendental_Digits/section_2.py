from manim import *

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
        lecture_lines = [
            "Conservation of momentum defines the physics.",
            "Elastic collisions preserve kinetic energy.",
            "We visualize this in phase space."
        ]
        self.setup_layout("Prerequisite: Conservation Laws", lecture_lines)
        
        # Elements
        eq = MathTex(r"m_1 v_1 + m_2 v_2 = \text{const}", font_size=36)
        billiard_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/billiard.svg")
        
        # === Animation for Lecture Line 1 ===
        self.place_in_area(eq, 'A1', 'A6', scale_factor=1.0)
        self.play(FadeIn(eq))
        self.place_at_grid(billiard_icon, 'B3', scale_factor=0.5)
        self.play(FadeIn(billiard_icon))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight m and v
        eq_m1 = eq[0][0:2]
        eq_v1 = eq[0][3:5]
        eq_m2 = eq[0][6:8]
        eq_v2 = eq[0][9:11]
        
        self.play(
            eq_m1.animate.set_color("#FFFF00"),
            eq_m2.animate.set_color("#FFFF00"),
            eq_v1.animate.set_color("#00FFFF"),
            eq_v2.animate.set_color("#00FFFF"),
            self.lecture[1].animate.set_color("#FFFF00")
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Add visual representation
        circle = Circle(radius=0.5, color=BLUE).set_fill(BLUE, opacity=0.5)
        self.place_at_grid(circle, 'E5', scale_factor=0.5)
        self.play(Create(circle), self.lecture[2].animate.set_color("#00FF00"))
        # Animate total momentum conserved with icon
        self.play(billiard_icon.animate.shift(RIGHT * 1))
        self.play(billiard_icon.animate.shift(LEFT * 1))
        self.wait(2)
