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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Visualizing the Tangent Line", [
            "Implicit derivative gives the tangent slope.", 
            "Slope depends on both x and y.", 
            "Tangent line visualizes the rate of change."
        ])
        
        # Assets
        graph_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/graph.svg")
        
        # Objects
        axes = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_tip": False}).scale(0.5)
        curve = ImplicitFunction(lambda x, y: x**2 + y**2 - 2, color=BLUE)
        point = Dot(color=YELLOW).move_to(curve.point_from_proportion(0.125)) 
        tangent = Line(start=ORIGIN, end=RIGHT*2, color=RED).rotate(PI/4).move_to(point)
        
        # Positioning
        self.place_in_area(axes, "B3", "E6", scale_factor=0.5)
        self.place_at_grid(graph_icon, "A3", scale_factor=0.3)
        curve.move_to(axes.c2p(0, 0))
        point.move_to(axes.c2p(1, 1))
        
        slope_label = MathTex("m = dy/dx", color=GREEN).scale(0.8)
        equation_label = MathTex("y - y_0 = m(x - x_0)", color=ORANGE).scale(0.6)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(axes), FadeIn(graph_icon), Create(curve), FadeIn(point))
        self.lecture[0].set_color(GREEN)
        self.play(Create(tangent))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GREEN)
        self.place_at_grid(slope_label, "B4", scale_factor=0.7)
        self.play(Write(slope_label))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN)
        self.place_at_grid(equation_label, "E4", scale_factor=0.7)
        self.play(Write(equation_label))
        
        self.wait(2)
