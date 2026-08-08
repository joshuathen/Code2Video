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
        self.setup_layout("Prerequisite Review: Graphical Slope vs. Area", 
                         ["Derivatives represent the slope of the tangent.", 
                          "Integrals represent the area under the curve.", 
                          "Both describe the same geometric function differently."])
        
        # Create axes and curve
        axes = Axes(x_range=[0, 3], y_range=[0, 3], axis_config={"include_numbers": False}).scale(0.5)
        curve = axes.plot(lambda x: 0.5 * x**2 + 0.5, color=WHITE)
        
        graph_group = VGroup(axes, curve)
        self.place_in_area(graph_group, 'C2', 'F6', scale_factor=0.6)
        
        # Tangent elements
        tangent_line = Line(start=LEFT, end=RIGHT, color="#FF4500").scale(0.3)
        tangent_line.rotate(PI/4)
        tangent_line.move_to(axes.c2p(1.5, 0.5 * 1.5**2 + 0.5))
        slope_label = Text("Slope", color="#FF4500", font_size=20)
        slope_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg").scale(0.2)
        slope_group = VGroup(slope_label, slope_icon).arrange(RIGHT, buff=0.1)
        self.place_at_grid(slope_group, 'B4')
        
        # Area elements
        area = axes.get_area(curve, x_range=[0, 2], color="#32CD32", opacity=0.5)
        area_label = Text("Area", color="#32CD32", font_size=20)
        area_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg").scale(0.2)
        area_group = VGroup(area_label, area_icon).arrange(RIGHT, buff=0.1)
        self.place_at_grid(area_group, 'B5')

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF4500")
        self.play(Create(tangent_line), FadeIn(slope_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#32CD32")
        self.play(FadeIn(area), FadeIn(area_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        self.play(
            FadeOut(tangent_line),
            FadeOut(area),
            FadeOut(slope_group),
            FadeOut(area_group),
            FadeOut(graph_group)
        )
        self.wait(1)
