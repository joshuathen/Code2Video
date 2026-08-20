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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["We express time as an integral.", "Time depends on path length and velocity.", "Calculus of variations finds the optimal curve."]
        self.setup_layout("The Mathematical Formulation", lecture_lines)
        
        # Animations
        # === Animation for Lecture Line 1 ===
        time_integral = MathTex("T = \\int \\frac{ds}{v}", color=WHITE)
        self.place_in_area(time_integral, 'A3', 'B6', scale_factor=1.0)
        
        # Asset reference: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg
        # Note: Using a placeholder Mobject as none.svg might be empty/non-functional
        icon1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg") if False else Dot(color=BLACK)
        
        self.play(Write(time_integral))
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        ds_sub = MathTex("ds = \\sqrt{dx^2 + dy^2}", color="#FF0000")
        v_sub = MathTex("v = \\sqrt{2gy}", color="#00FF00")
        sub_group = VGroup(ds_sub, v_sub).arrange(DOWN, buff=0.5)
        
        self.place_in_area(sub_group, 'C3', 'D6', scale_factor=0.9)
        
        self.play(Write(ds_sub))
        self.play(Write(v_sub))
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        functional = MathTex("J(y) = \\int \\sqrt{\\frac{1 + y'^2}{2gy}} dx", color="#FFFF00")
        label = Text("Time functional", font_size=20, color=WHITE)
        func_group = VGroup(functional, label).arrange(DOWN, buff=0.3)
        
        # Asset reference: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg
        icon2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg") if False else Dot(color=BLACK)

        self.place_in_area(func_group, 'E2', 'F6', scale_factor=0.8)
        
        self.play(Write(functional))
        self.play(FadeIn(label))
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        self.wait(2)
