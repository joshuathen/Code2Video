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
        lecture_lines = [
            "Concavity shows how a curve bends.",
            "Positive second derivative means concave up.",
            "Negative second derivative means concave down.",
            "This curvature visually identifies acceleration changes.",
            "Observe how the slope increases or decreases."
        ]
        self.setup_layout("Visualizing Concavity and Curvature", lecture_lines)
        
        # Axis and Curve definitions
        axes = Axes(x_range=[-2, 2], y_range=[-1, 3], axis_config={"include_tip": False})
        self.place_in_area(axes, "C3", "E6", scale_factor=0.55)
        
        # Concave up curve (f(x) = x^2)
        curve_up = axes.plot(lambda x: x**2 + 0.5, x_range=[-1.2, 1.2], color="#00FF00")
        
        # Concave down curve (f(x) = -x^2 + 2)
        curve_down = axes.plot(lambda x: -x**2 + 2, x_range=[-1.2, 1.2], color="#FF0000")
        
        # Icons
        bowl = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bowl.svg")
        dome = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dome.svg")
        coaster = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rollercoaster.svg")
        
        parabolas = VGroup(curve_up, curve_down)
        self.place_in_area(parabolas, "B4", "E6", scale_factor=0.45)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.add(axes)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        self.place_at_grid(bowl, "B2", scale_factor=0.3)
        self.play(Create(curve_up), FadeIn(bowl))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF0000"))
        self.place_at_grid(dome, "B5", scale_factor=0.3)
        self.play(Create(curve_down), FadeIn(dome))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(YELLOW))
        inflection_dot = Dot(axes.c2p(0, 1.25), color=WHITE)
        self.play(Create(inflection_dot))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(YELLOW))
        self.place_at_grid(coaster, "E2", scale_factor=0.4)
        self.play(FadeIn(coaster))
        self.wait(2)
