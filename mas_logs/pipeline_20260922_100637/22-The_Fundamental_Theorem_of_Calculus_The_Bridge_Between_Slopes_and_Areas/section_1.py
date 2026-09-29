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
        self.setup_layout("Prerequisite Review: The Two Pillars", [
            "Differentiation finds instantaneous slope.",
            "Integration calculates accumulated area.",
            "These seem like opposites."
        ])
        
        # Assets
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg")
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        
        # Elements
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: 0.5 * x**2, x_range=[0, 4], color=WHITE)
        self.place_in_area(axes, 'B2', 'F6', scale_factor=0.5)
        curve.scale(0.5).move_to(axes.get_center())
        
        dot = Dot(curve.point_from_proportion(0.5), color="#FF5733")
        tangent_line = Line(start=LEFT*0.5, end=RIGHT*0.5, color="#FF5733").move_to(dot)
        area = axes.get_area(curve, x_range=[0, 2], color="#33FF57", opacity=0.5)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF5733")
        self.add(axes, curve)
        self.place_at_grid(ruler, 'D1', scale_factor=0.3)
        self.play(FadeIn(dot), GrowFromCenter(tangent_line), FadeIn(ruler))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#33FF57")
        self.place_at_grid(protractor, 'D2', scale_factor=0.3)
        self.play(FadeIn(area), FadeIn(protractor))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#3357FF")
        self.place_at_grid(compass, 'D3', scale_factor=0.3)
        group = VGroup(dot, tangent_line, area)
        self.play(group.animate.set_color("#3357FF"), FadeIn(compass))
        self.wait(2)
