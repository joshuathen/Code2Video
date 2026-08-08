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
        lines = ["Velocity: From total distance to exact instant.", 
                 "Imagine a cheetah lunging at a gazelle.", 
                 "We measure speed at that precise point.", 
                 "Average speed becomes instantaneous speed.", 
                 "Calculus reveals the truth of the moment."]
        self.setup_layout("Intuitive Hook: The Velocity Analogy", lines)
        
        # Animation Elements
        gauge = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/gauge.svg")
        needle = Line(ORIGIN, UP*0.8, color="#FF5733")
        gauge_group = VGroup(gauge, needle)
        
        axes = Axes(x_range=[0, 5, 1], y_range=[0, 5, 1], x_length=4, y_length=3).scale(0.5)
        curve = axes.plot(lambda x: 0.2 * x**2, color="#33FF57")
        dot = Dot(axes.c2p(2, 0.8), color="#F3FF33")
        slope_line = TangentLine(curve, alpha=0.4, length=1.5, color="#F3FF33")
        cheetah = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg")
        
        # === Animation for Lecture Line 1 ===
        self.place_in_area(gauge_group, 'A4', 'C6', scale_factor=0.65)
        self.play(FadeIn(gauge_group))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        self.play(Rotate(needle, angle=-PI/2, about_point=gauge.get_center(), rate_func=linear))
        self.lecture[1].set_color("#FF5733")

        # === Animation for Lecture Line 3 ===
        self.play(FadeOut(gauge_group))
        self.place_in_area(axes, 'D4', 'F6', scale_factor=0.7)
        self.play(FadeIn(axes), FadeIn(curve))
        self.lecture[2].set_color("#33FF57")

        # === Animation for Lecture Line 4 ===
        self.play(Create(dot), Create(slope_line))
        self.lecture[3].set_color("#F3FF33")

        # === Animation for Lecture Line 5 ===
        self.place_at_grid(cheetah, 'E5', scale_factor=0.5)
        self.play(FadeIn(cheetah), slope_line.animate.scale(1.2))
        self.lecture[4].set_color("#FFFFFF")
