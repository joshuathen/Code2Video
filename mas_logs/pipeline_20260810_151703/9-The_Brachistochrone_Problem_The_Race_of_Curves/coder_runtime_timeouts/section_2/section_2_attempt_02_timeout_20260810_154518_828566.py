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
        self.setup_layout("Prerequisite Intuition: Energy and Velocity", 
                         ["Energy dictates speed, gravity drives descent.", 
                          "Gaining velocity early is crucial.", 
                          "Curvature helps build speed quickly."])
        
        # Assets
        ramp = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ramp.svg", color="#FFD700")
        ball = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg")
        
        # Kinetic Energy formula
        # MathTex: K = 0.5 * m * v^2
        formula = MathTex("K", "=", "0.5", "\\cdot", "m", "\\cdot", "v^2", color="#00FFFF")
        self.place_at_grid(formula, 'B2', scale_factor=1.2)
        
        # Energy Bar
        energy_bar_bg = Rectangle(height=0.3, width=3, color=WHITE, fill_opacity=0.2)
        energy_bar = Rectangle(height=0.3, width=0, color="#FFFF00", fill_opacity=0.8)
        energy_bar.align_to(energy_bar_bg, LEFT)
        bar_group = VGroup(energy_bar_bg, energy_bar)
        self.place_at_grid(bar_group, 'D2', scale_factor=1.0)
        
        velocity_val = ValueTracker(0)
        energy_bar.add_updater(lambda m: m.set_width(3 * (velocity_val.get_value() / 5), stretch=True))

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        self.play(Write(formula))
        self.place_at_grid(ramp, 'E3', scale_factor=0.5)
        self.play(Create(ramp), Create(bar_group))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFD700")
        self.place_at_grid(ball, 'D3', scale_factor=0.3)
        self.play(FadeIn(ball), velocity_val.animate.set_value(5), run_time=2)
        self.play(Indicate(formula[6])) # Flash velocity term v^2

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFD700")
        self.wait(1)
        self.play(FadeOut(formula), FadeOut(bar_group), FadeOut(ramp), FadeOut(ball))
