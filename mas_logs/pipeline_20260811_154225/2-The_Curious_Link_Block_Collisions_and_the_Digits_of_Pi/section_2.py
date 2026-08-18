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
        self.setup_layout(
            "Prerequisite Physics: Conservation Laws", 
            ["Elastic collisions conserve both momentum and kinetic energy.", 
             "These laws map physical states to geometric rotations.", 
             "Collisions behave like reflections in a velocity plane."]
        )
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg
        ball_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg")
        self.place_at_grid(ball_icon, 'A6', scale_factor=0.3)
        self.add(ball_icon)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        
        energy_formula = MathTex(r"E_{\text{initial}} = E_{\text{final}}", color="#FFFF00")
        self.place_in_area(energy_formula, 'A2', 'C4', scale_factor=0.9)
        self.play(Write(energy_formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/billiard.svg
        billiard_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/billiard.svg")
        
        # Represent energy bars for state shift
        energy_bar_bg = Rectangle(height=0.5, width=3, color=GRAY, fill_opacity=0.3)
        energy_bar = Rectangle(height=0.5, width=3, color="#00FFFF", fill_opacity=0.8)
        energy_group = VGroup(energy_bar_bg, energy_bar, billiard_icon)
        energy_group.arrange(DOWN)
        self.place_in_area(energy_group, 'D2', 'E4', scale_factor=1.1)
        self.add(energy_group)
        
        # Animate shifting state
        self.play(energy_bar.animate.stretch_to_fit_width(1.5), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        
        # Placeholder for collision reflection visualization
        dots = VGroup(Dot(color=WHITE), Dot(color=WHITE)).arrange(RIGHT, buff=0.5)
        self.place_at_grid(dots, 'E3', scale_factor=1.0)
        self.play(FadeIn(dots))
        self.play(Rotate(dots, angle=PI/2, about_point=self.grid['E3']))
        self.wait(2)
