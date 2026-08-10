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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Concept: Beyond Single Variables", [
            "ODE: One fly moving in a line.",
            "PDE: Ripples spreading across a pond.",
            "PDEs track changes in multiple dimensions."
        ])

        # === Animation for Lecture Line 1 ===
        fly_dot = Dot(color="#FFFFFF")
        fly_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/fly.svg")
        fly = VGroup(fly_dot, fly_img).arrange(RIGHT, buff=0.1)
        self.place_at_grid(fly, 'A5', scale_factor=0.8)
        self.play(FadeIn(fly))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        axes = ThreeDAxes(x_range=[-2, 2], y_range=[-2, 2], z_range=[-1, 1])
        ripple = axes.plot_surface(
            lambda u, v: 0.5 * np.sin(np.sqrt(u**2 + v**2) * 3),
            u_range=[-2, 2], v_range=[-2, 2],
            color="#00FF00"
        ).set_opacity(0.8)
        pond_group = VGroup(axes, ripple)
        self.place_in_area(pond_group, 'B4', 'E6', scale_factor=0.5)
        
        self.play(FadeOut(fly), FadeIn(pond_group))
        self.lecture[1].set_color("#00FF00")

        # === Animation for Lecture Line 3 ===
        pond_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pond.svg")
        self.place_at_grid(pond_icon, 'F1', scale_factor=0.5)
        
        z_axis_highlight = Line(axes.c2p(0, 0, -1), axes.c2p(0, 0, 1), color="#FF00FF", stroke_width=4)
        pond_group.add(z_axis_highlight)
        
        cross_section = axes.plot(lambda x: 0.5 * np.sin(np.abs(x) * 3), x_range=[-2, 2], color="#FFFF00", stroke_width=4)
        formula_label = MathTex("f(x, y_0)", color=WHITE)
        self.place_at_grid(formula_label, 'B3', scale_factor=0.7)
        
        self.play(FadeIn(pond_icon))
        self.play(FadeIn(z_axis_highlight))
        self.play(Create(cross_section))
        self.play(Write(formula_label))
        self.lecture[2].set_color("#FF00FF")
        
        self.wait(2)
